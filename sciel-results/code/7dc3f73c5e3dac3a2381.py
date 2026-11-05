import itertools
import time
import sys
import random

def _standard_superperm(n):
    """Generate standard superpermutation for n using recursive construction."""
    if n == 1:
        return [1]
    if n == 2:
        return [1, 2, 1]
    if n == 3:
        return [1, 2, 3, 1, 2, 1, 3, 2, 1]
    
    prev = _standard_superperm(n - 1)
    
    seen = set()
    order = []
    for i in range(len(prev) - (n - 1) + 1):
        window = tuple(prev[i:i + n - 1])
        if window not in seen:
            seen.add(window)
            if sorted(window) == list(range(1, n)):
                order.append(window)
    
    new = []
    for perm in order:
        block = list(perm) + [n] + list(perm)
        max_ov = min(len(new), len(block))
        overlap = 0
        for ov in range(max_ov, -1, -1):
            if ov == 0 or new[len(new) - ov:] == block[:ov]:
                overlap = ov
                break
        new.extend(block[overlap:])
    
    return new

def _verify_superpermutation(word, n):
    """Check if word contains all permutations of 1..n as consecutive subsequences."""
    all_perms = set(itertools.permutations(range(1, n + 1)))
    found = set()
    for i in range(len(word) - n + 1):
        p = tuple(word[i:i + n])
        if p in all_perms:
            found.add(p)
        if len(found) == len(all_perms):
            return True
    return len(found) == len(all_perms)

def _construct_from_order(order, n):
    """
    Construct superpermutation for n from the given order of (n-1)-permutations.
    Returns the word as a list of ints.
    """
    new = []
    for perm in order:
        block = list(perm) + [n] + list(perm)
        max_ov = min(len(new), len(block))
        overlap = 0
        for ov in range(max_ov, -1, -1):
            if ov == 0 or new[len(new) - ov:] == block[:ov]:
                overlap = ov
                break
        new.extend(block[overlap:])
    return new

def _extract_order_from_word(word, n):
    """Extract the order of (n-1)-permutations from a superpermutation for n."""
    seen = set()
    order = []
    for i in range(len(word) - (n - 1) + 1):
        window = tuple(word[i:i + n - 1])
        if window not in seen:
            seen.add(window)
            if sorted(window) == list(range(1, n)):
                order.append(window)
    return order

def _compute_length_from_order(order, n):
    """
    Compute the exact length of the superpermutation built from the given order.
    Uses incremental string building with overlap detection.
    """
    total = 0
    current = []
    for perm in order:
        block = list(perm) + [n] + list(perm)
        max_ov = min(len(current), len(block))
        overlap = 0
        for ov in range(max_ov, -1, -1):
            if ov == 0 or current[len(current) - ov:] == block[:ov]:
                overlap = ov
                break
        total += len(block) - overlap
        current.extend(block[overlap:])
        # Keep current bounded for speed - we only need the tail
        if len(current) > 2 * n + 10:
            current = current[-(2 * n + 10):]
    return total

def _compute_overlap_fast(p_i, p_j):
    """Compute overlap between two (n-1)-permutations efficiently."""
    max_ov = min(len(p_i), len(p_j)) - 1
    for ov in range(max_ov, -1, -1):
        if p_i[len(p_i)-ov:] == p_j[:ov]:
            return ov
    return 0

def _greedy_nearest_neighbor_order(perms, n, deadline=None):
    """
    Use a Greedy Nearest Neighbor heuristic with multiple starts and local search
    to find a good ordering of (n-1)-permutations.
    Optimized for speed: precompute overlap matrix, use efficient data structures.
    """
    num_perms = len(perms)
    if num_perms <= 1:
        return perms
    
    # Precompute overlap matrix using efficient method
    # For (n-1)-perms, max overlap is n-2
    max_ov_possible = n - 2
    overlap_matrix = [[0] * num_perms for _ in range(num_perms)]
    
    for i in range(num_perms):
        p_i = perms[i]
        for j in range(num_perms):
            if i == j:
                continue
            p_j = perms[j]
            # Check from max possible overlap down
            for ov in range(max_ov_possible, -1, -1):
                if ov == 0:
                    break
                if p_i[len(p_i)-ov:] == p_j[:ov]:
                    overlap_matrix[i][j] = ov
                    break
    
    def _greedy_from_start(start_idx):
        """Run greedy NN from a given starting permutation."""
        visited = [False] * num_perms
        order_idx = []
        
        current = start_idx
        visited[current] = True
        order_idx.append(current)
        
        for _ in range(num_perms - 1):
            best_next = -1
            best_overlap = -1
            
            for next_idx in range(num_perms):
                if not visited[next_idx]:
                    ov = overlap_matrix[current][next_idx]
                    if ov > best_overlap:
                        best_overlap = ov
                        best_next = next_idx
            
            if best_next == -1:
                break
                
            visited[best_next] = True
            order_idx.append(best_next)
            current = best_next
            
        return order_idx
    
    def _order_cost(order_idx):
        """Compute total overlap cost (higher is better) for an ordering."""
        total = 0
        for k in range(len(order_idx) - 1):
            total += overlap_matrix[order_idx[k]][order_idx[k+1]]
        return total
    
    def _two_opt_fast(order_idx, deadline_local):
        """Apply 2-opt local search with incremental cost updates."""
        best_order = list(order_idx)
        best_cost = _order_cost(best_order)
        improved = True
        max_iters = 200
        
        while improved and max_iters > 0:
            improved = False
            max_iters -= 1
            
            for i in range(len(best_order) - 1):
                if deadline_local and time.time() > deadline_local:
                    return best_order
                for j in range(i + 2, len(best_order)):
                    # Compute delta cost for reversing [i+1..j]
                    # Old edges: (i, i+1) and (j, j+1) if j < len-1
                    # New edges: (i, j) and (j-1, i+1) if j-1 >= i+1
                    old_cost = overlap_matrix[best_order[i]][best_order[i+1]]
                    if j < len(best_order) - 1:
                        old_cost += overlap_matrix[best_order[j]][best_order[j+1]]
                    
                    new_cost = overlap_matrix[best_order[i]][best_order[j]]
                    if j > i + 1:
                        new_cost += overlap_matrix[best_order[j-1]][best_order[i+1]]
                    
                    if new_cost > old_cost:
                        best_order[i+1:j+1] = best_order[i+1:j+1][::-1]
                        best_cost += new_cost - old_cost
                        improved = True
        
        return best_order
    
    def _or_opt(order_idx, deadline_local):
        """Or-opt: move a segment of 1-3 elements to a new position."""
        best_order = list(order_idx)
        best_cost = _order_cost(best_order)
        improved = True
        max_iters = 100
        
        while improved and max_iters > 0:
            improved = False
            max_iters -= 1
            
            for seg_len in [1, 2, 3]:
                if deadline_local and time.time() > deadline_local:
                    return best_order
                for i in range(len(best_order) - seg_len):
                    if deadline_local and time.time() > deadline_local:
                        return best_order
                    seg = best_order[i:i+seg_len]
                    # Remove segment
                    for dest in range(len(best_order) - seg_len + 1):
                        if dest <= i and dest + seg_len > i:
                            continue
                        if dest == i:
                            continue
                        # Compute cost change
                        old_edges = []
                        new_edges = []
                        
                        # Edge before segment
                        if i > 0:
                            old_edges.append(overlap_matrix[best_order[i-1]][best_order[i]])
                        # Edge after segment
                        if i + seg_len < len(best_order):
                            old_edges.append(overlap_matrix[best_order[i+seg_len-1]][best_order[i+seg_len]])
                        
                        # Determine insert position
                        actual_dest = dest if dest < i else dest - seg_len
                        # Edge before insert
                        if actual_dest > 0:
                            new_edges.append(overlap_matrix[best_order[actual_dest-1]][seg[0]])
                        else:
                            new_edges.append(0)
                        # Edge after insert
                        if actual_dest + seg_len < len(best_order) - seg_len:
                            new_edges.append(overlap_matrix[seg[-1]][best_order[actual_dest+seg_len]])
                        else:
                            new_edges.append(0)
                        
                        old_total = sum(old_edges)
                        new_total = sum(new_edges)
                        
                        if new_total > old_total:
                            # Perform the move
                            new_order = best_order[:i] + best_order[i+seg_len:]
                            insert_pos = dest if dest < i else dest - seg_len
                            new_order = new_order[:insert_pos] + seg + new_order[insert_pos:]
                            best_order = new_order
                            best_cost += new_total - old_total
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        
        return best_order
    
    # Try multiple starting points
    start_indices = [0, num_perms // 2, num_perms - 1]
    # Add some random starts
    random.seed(42)
    for _ in range(5):
        start_indices.append(random.randint(0, num_perms - 1))
    
    best_order_idx = None
    best_cost = -1
    
    for si in start_indices:
        if deadline and time.time() > deadline - 30:
            break
        order_idx = _greedy_from_start(si)
        cost = _order_cost(order_idx)
        if cost > best_cost:
            best_cost = cost
            best_order_idx = order_idx
    
    # Apply local search refinement
    if best_order_idx is not None:
        if deadline:
            deadline_2opt = deadline - 15
        else:
            deadline_2opt = None
        best_order_idx = _two_opt_fast(best_order_idx, deadline_2opt)
        
        if deadline:
            deadline_oropt = deadline - 5
        else:
            deadline_oropt = None
        best_order_idx = _or_opt(best_order_idx, deadline_oropt)
    
    # Convert to permutation tuples
    return [perms[i] for i in best_order_idx]

def _delete_pass(word, n, deadline):
    """
    Character-level deletion pass with multiple sweeps.
    A character can be removed if no permutation window that is unique is destroyed.
    """
    S = list(word)
    
    def _build_counts(S):
        counts = {}
        for i in range(len(S) - n + 1):
            p = tuple(S[i:i+n])
            counts[p] = counts.get(p, 0) + 1
        return counts
    
    counts = _build_counts(S)
    
    # Multiple passes until no more deletions or time runs out
    while time.time() < deadline:
        made_removal = False
        
        # Forward pass
        i = 0
        while i < len(S):
            if time.time() > deadline:
                break
            
            # Old windows that include position i: indices i-n+1 to i
            start_old = max(0, i - n + 1)
            end_old = min(i, len(S) - n)
            
            valid = True
            old_perms_to_dec = []
            for j in range(start_old, end_old + 1):
                p = tuple(S[j:j+n])
                if counts.get(p, 0) == 1:
                    valid = False
                    break
                old_perms_to_dec.append(p)
            
            if not valid:
                i += 1
                continue
            
            # Perform removal
            for p in old_perms_to_dec:
                counts[p] -= 1
                if counts[p] == 0:
                    del counts[p]
            
            # New windows formed by straddling the gap
            start_new = max(0, i - n + 1)
            end_new = min(i - 1, len(S) - 2 - n + 1)
            for k in range(start_new, end_new + 1):
                window_chars = []
                for idx in range(k, k+n):
                    s_idx = idx if idx < i else idx + 1
                    window_chars.append(S[s_idx])
                p_new = tuple(window_chars)
                counts[p_new] = counts.get(p_new, 0) + 1
            
            S.pop(i)
            made_removal = True
            # Don't increment i since we removed an element
        
        # Backward pass
        i = len(S) - 1
        while i >= 0:
            if time.time() > deadline:
                break
            
            start_old = max(0, i - n + 1)
            end_old = min(i, len(S) - n)
            
            valid = True
            old_perms_to_dec = []
            for j in range(start_old, end_old + 1):
                p = tuple(S[j:j+n])
                if counts.get(p, 0) == 1:
                    valid = False
                    break
                old_perms_to_dec.append(p)
            
            if not valid:
                i -= 1
                continue
            
            for p in old_perms_to_dec:
                counts[p] -= 1
                if counts[p] == 0:
                    del counts[p]
            
            start_new = max(0, i - n + 1)
            end_new = min(i - 1, len(S) - 2 - n + 1)
            for k in range(start_new, end_new + 1):
                window_chars = []
                for idx in range(k, k+n):
                    s_idx = idx if idx < i else idx + 1
                    window_chars.append(S[s_idx])
                p_new = tuple(window_chars)
                counts[p_new] = counts.get(p_new, 0) + 1
            
            S.pop(i)
            made_removal = True
            i -= 1
        
        if not made_removal:
            break
    
    return S

def construct(n):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    # For n <= 5, return standard (optimal for n<=3)
    if n <= 5:
        word = _standard_superperm(n)
        # Apply deletion pass for potential improvement
        if time.time() < deadline:
            word = _delete_pass(word, n, deadline)
        return "".join(map(str, word))
    
    # For n >= 6: 
    # 1. Generate all (n-1)-permutations
    perms = list(itertools.permutations(range(1, n)))
    
    # 2. Use Greedy Nearest Neighbor with multiple starts + local search
    if time.time() < deadline - 20:
        deadline_greedy = deadline - 20
        order = _greedy_nearest_neighbor_order(perms, n, deadline_greedy)
    else:
        # Fallback to standard order if no time
        base = _standard_superperm(n - 1)
        order = _extract_order_from_word(base, n - 1)
        
    # 3. Construct the final superpermutation
    word = _construct_from_order(order, n)
    
    # 4. Character-level deletion pass (multiple sweeps)
    if time.time() < deadline - 5:
        deadline_del = deadline - 5
        word = _delete_pass(word, n, deadline_del)
        # Second deletion pass if time permits
        if time.time() < deadline - 3:
            word = _delete_pass(word, n, deadline - 3)
    
    # Verify
    if not _verify_superpermutation(word, n):
        # Fallback to standard
        word = _standard_superperm(n)
    
    return "".join(map(str, word))

# Quick self-test
if __name__ == "__main__":
    for n in range(1, 6):
        w = construct(n)
        valid = _verify_superpermutation(list(map(int, w)), n)
        print(f"n={n}, len={len(w)}, valid={valid}")
    
    for n in [6, 7]:
        start = time.time()
        w = construct(n)
        elapsed = time.time() - start
        valid = _verify_superpermutation(list(map(int, w)), n)
        print(f"n={n}, len={len(w)}, valid={valid}, time={elapsed:.2f}s")
