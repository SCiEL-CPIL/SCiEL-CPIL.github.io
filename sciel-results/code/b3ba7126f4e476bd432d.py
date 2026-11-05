"""Phylogenetic topology recovery: vectorized NJ + fast NNI refinement under JC."""
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 255.0


def _jc_distance_matrix(sequences, labels):
    """Compute pairwise Jukes-Cantor corrected distances using NumPy."""
    n = len(labels)
    L = len(sequences[labels[0]])

    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seq_arr = np.array([[mapping[c] for c in sequences[lab]] for lab in labels], dtype=np.int8)

    diff = (seq_arr[:, np.newaxis, :] != seq_arr[np.newaxis, :, :]).sum(axis=-1)

    p = diff.astype(np.float64) / L
    p = np.clip(p, 0, 0.7499)
    D = -0.75 * np.log(1.0 - (4.0 / 3.0) * p)
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) * 0.5

    return D


def _nj_build(D, labels):
    """
    Vectorized Neighbor Joining.
    Returns Newick string.
    """
    n = len(labels)
    if n <= 1:
        return labels[0] + ";" if labels else ""
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]});"

    active = list(range(n))
    newick = {i: labels[i] for i in range(n)}

    # Use a dict for distances to avoid rebuilding matrices
    dist = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                dist[(i, j)] = D[i, j]

    next_id = n

    while len(active) > 3:
        m = len(active)
        if time.time() > _DEADLINE - 10.0:
            break

        # Compute Q criterion vectorized
        r = np.zeros(m)
        for i, a in enumerate(active):
            for j, b in enumerate(active):
                if i != j:
                    r[i] += dist[(a, b)]

        best_val = float('inf')
        best_a, best_b = active[0], active[1]

        for i in range(m):
            a = active[i]
            for j in range(i + 1, m):
                b = active[j]
                q_val = (m - 2) * dist[(a, b)] - r[i] - r[j]
                if q_val < best_val:
                    best_val = q_val
                    best_a, best_b = a, b

        a, b = best_a, best_b

        new_id_val = next_id
        next_id += 1
        new_newick = f"({newick[a]},{newick[b]})"

        ab_dist = dist[(a, b)]
        for x in active:
            if x != a and x != b:
                new_d = 0.5 * (dist[(a, x)] + dist[(b, x)] - ab_dist)
                dist[(new_id_val, x)] = new_d
                dist[(x, new_id_val)] = new_d

        active = [x for x in active if x != a and x != b]
        active.append(new_id_val)
        newick[new_id_val] = new_newick

    if len(active) == 3:
        a, b, c = active
        newick_str = f"({newick[a]},({newick[b]},{newick[c]}));"
    else:
        newick_str = "(" + ",".join(newick[x] for x in active) + ");"

    return newick_str


def _parse_newick_leaves(newick_str, label_set):
    """Parse a Newick string and return a list of leaf labels."""
    leaves = []
    i = 0
    n = len(newick_str)

    while i < n:
        ch = newick_str[i]
        if ch == '(':
            depth = 0
            j = i
            while j < n:
                if newick_str[j] == '(':
                    depth += 1
                elif newick_str[j] == ')':
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            subtree = newick_str[i+1:j]
            sub_leaves = _parse_newick_leaves(subtree, label_set)
            leaves.extend(sub_leaves)
            i = j + 1
            if i < n and newick_str[i] == ':':
                i += 1
                while i < n and newick_str[i] not in ',;':
                    i += 1
            elif i < n and newick_str[i] == ',':
                i += 1
        elif ch in ',;':
            i += 1
        else:
            j = i
            while j < n and newick_str[j] not in '(),;:':
                j += 1
            label = newick_str[i:j]
            if label in label_set:
                leaves.append(label)
            i = j

    return leaves


def _nni_refine(D, newick_str, labels, label_set):
    """
    Fast NNI refinement using quartet scores.
    Precompute leaf indices for each subtree to speed up scoring.
    """
    n = len(labels)
    if n < 4:
        return newick_str

    label_to_idx = {lab: i for i, lab in enumerate(labels)}

    def compute_quartet_score(leaves1, leaves2, leaves3, leaves4):
        """Compute sum of Q for topology (1,2)|(3,4)."""
        total = 0.0
        idx1 = [label_to_idx[x] for x in leaves1]
        idx2 = [label_to_idx[x] for x in leaves2]
        idx3 = [label_to_idx[x] for x in leaves3]
        idx4 = [label_to_idx[x] for x in leaves4]
        
        for i1 in idx1:
            for i2 in idx2:
                d12 = D[i1, i2]
                for i3 in idx3:
                    d13 = D[i1, i3]
                    d23 = D[i2, i3]
                    for i4 in idx4:
                        d34 = D[i3, i4]
                        d14 = D[i1, i4]
                        d24 = D[i2, i4]
                        total += d12 + d34 - d13 - d14 - d23 - d24
        return total

    max_rounds = 30
    for rnd in range(max_rounds):
        if time.time() > _DEADLINE - 5.0:
            break

        improved = False
        newick = newick_str

        # Find internal edges: pattern "),(" 
        positions = []
        for pos in range(len(newick) - 3):
            if newick[pos:pos+2] == ")," and newick[pos+2:pos+3] == "(":
                positions.append(pos)

        for pos in positions:
            if time.time() > _DEADLINE - 3.0:
                break

            # Extract left subtree (ends at pos)
            left_end = pos
            depth = 0
            left_start = left_end
            for k in range(left_end, -1, -1):
                if newick[k] == ')':
                    depth += 1
                elif newick[k] == '(':
                    depth -= 1
                    if depth == 0:
                        left_start = k
                        break

            # Extract right subtree (starts at pos+2)
            right_start = pos + 2
            depth = 0
            right_end = right_start
            for k in range(right_start, len(newick)):
                if newick[k] == '(':
                    depth += 1
                elif newick[k] == ')':
                    depth -= 1
                    if depth == 0:
                        right_end = k
                        break

            left_content = newick[left_start+1:left_end]
            right_content = newick[right_start+1:right_end]

            # Split left into two subtrees
            depth = 0
            comma_pos = -1
            for k in range(len(left_content)):
                if left_content[k] == '(':
                    depth += 1
                elif left_content[k] == ')':
                    depth -= 1
                elif left_content[k] == ',' and depth == 0:
                    comma_pos = k
                    break

            if comma_pos == -1:
                continue

            left_a = left_content[:comma_pos]
            left_b = left_content[comma_pos+1:]

            # Split right into two subtrees
            depth = 0
            comma_pos2 = -1
            for k in range(len(right_content)):
                if right_content[k] == '(':
                    depth += 1
                elif right_content[k] == ')':
                    depth -= 1
                elif right_content[k] == ',' and depth == 0:
                    comma_pos2 = k
                    break

            if comma_pos2 == -1:
                continue

            right_c = right_content[:comma_pos2]
            right_d = right_content[comma_pos2+1:]

            # Get leaf sets
            leaves_a = _parse_newick_leaves(left_a, label_set)
            leaves_b = _parse_newick_leaves(left_b, label_set)
            leaves_c = _parse_newick_leaves(right_c, label_set)
            leaves_d = _parse_newick_leaves(right_d, label_set)

            if not leaves_a or not leaves_b or not leaves_c or not leaves_d:
                continue

            # Skip if any subtree is too large (expensive)
            if len(leaves_a) * len(leaves_b) * len(leaves_c) * len(leaves_d) > 10000:
                continue

            # Current: (a,b)|(c,d)
            score_current = compute_quartet_score(leaves_a, leaves_b, leaves_c, leaves_d)
            # Alt1: (a,c)|(b,d)
            score_alt1 = compute_quartet_score(leaves_a, leaves_c, leaves_b, leaves_d)
            # Alt2: (a,d)|(b,c)
            score_alt2 = compute_quartet_score(leaves_a, leaves_d, leaves_b, leaves_c)

            best_score = min(score_current, score_alt1, score_alt2)

            if best_score < score_current - 1e-9:
                if best_score == score_alt1:
                    new_left = f"({left_a},{right_c})"
                    new_right = f"({left_b},{right_d})"
                else:
                    new_left = f"({left_a},{right_d})"
                    new_right = f"({left_b},{right_c})"
                new_segment = f"({new_left},{new_right})"

                old_segment = f"({left_content},{right_content})"
                if old_segment in newick:
                    newick = newick.replace(old_segment, new_segment, 1)
                    improved = True
                    break

        newick_str = newick
        if not improved:
            break

    return newick_str


def reconstruct(sequences):
    global _START_TIME, _DEADLINE

    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)

    if n <= 1:
        return labels[0] + ";" if labels else ""
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]});"

    try:
        D = _jc_distance_matrix(sequences, labels)
        newick = _nj_build(D, labels)

        if time.time() < _DEADLINE - 10.0:
            label_set = set(labels)
            newick = _nni_refine(D, newick, labels, label_set)

        if not newick.endswith(';'):
            newick += ';'

        return newick

    except Exception:
        return "(" + ",".join(labels) + ");"
