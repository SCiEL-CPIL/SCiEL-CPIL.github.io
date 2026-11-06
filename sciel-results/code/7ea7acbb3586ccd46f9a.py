import time

def construct():
    t0 = time.time()
    deadline = 80.0

    P = 101
    D = list(range(1, 11))

    # Precompute modular inverses
    inv = {x: pow(x, -1, P) for x in D}

    # Enumerate all non-empty subsets
    all_subsets = []
    for mask in range(1, 1 << 10):
        terms = []
        s = 0
        for x in range(1, 11):
            if (mask >> (x - 1)) & 1:
                terms.append(x)
                s = (s + inv[x]) % P
        size = len(terms)
        all_subsets.append((s, size, terms))

    # Find the maximum row length (L)
    max_len = 0
    for s, size, terms in all_subsets:
        max_len = max(max_len, size)

    # For each possible row length L from 1 to max_len,
    # try to find an assignment where every residue gets a subset of size <= L
    # and minimize the number of rows with size == L (to break ties)
    
    # Group subsets by residue
    from collections import defaultdict
    subsets_by_residue = defaultdict(list)
    for s, size, terms in all_subsets:
        subsets_by_residue[s].append((size, terms))

    # For each L, sort candidates for each residue by size (ascending)
    # We want to assign to each residue a subset of size <= L
    # Minimize the maximum size used
    
    # Try L from 1 upward, find if all 101 residues can be covered
    # For a given L, each residue needs at least one subset of size <= L
    best_L = None
    best_assignment = None

    for L in range(1, max_len + 1):
        # Check if every residue has at least one subset of size <= L
        all_covered = True
        candidates = {}
        for r in range(P):
            avail = [x for x in subsets_by_residue.get(r, []) if x[0] <= L]
            if not avail:
                all_covered = False
                break
            candidates[r] = avail
        if not all_covered:
            continue

        # Sort candidates for each residue: prefer smaller size
        # Then use a greedy assignment that minimizes the number of "large" (size==L) rows
        # For each residue, pick the candidate with smallest size
        # This is already optimal per-residue, but we need to check if it's a valid assignment
        # Since denominators can repeat across rows, there's no conflict between rows
        # So for each residue, just pick the smallest-size candidate
        
        assignment = {}
        num_large = 0
        for r in range(P):
            # Pick the candidate with smallest size
            best_candidate = min(candidates[r], key=lambda x: x[0])
            assignment[r] = best_candidate[1]
            if best_candidate[0] == L:
                num_large += 1

        # Verify
        valid = True
        for r in range(P):
            row = assignment[r]
            if len(row) == 0:
                valid = False
                break
            if len(set(row)) != len(row):
                valid = False
                break
            if not all(1 <= x <= 10 for x in row):
                valid = False
                break
            s = sum(inv[x] for x in row) % P
            if s != r:
                valid = False
                break

        if valid:
            if best_L is None or L < best_L:
                best_L = L
                best_assignment = assignment
            elif L == best_L and num_large < best_assignment_count:
                best_assignment = assignment
            break  # First valid L is the minimum

    # Build result
    result = [best_assignment[r] for r in range(P)]

    # Final verification
    for r in range(P):
        row = result[r]
        assert len(row) > 0, f"Row {r} is empty"
        assert len(set(row)) == len(row), f"Row {r} has duplicate denominators"
        assert all(1 <= x <= 10 for x in row), f"Row {r} has invalid denominator"
        s = sum(inv[x] for x in row) % P
        assert s == r, f"Row {r}: expected residue {r}, got {s}"

    return result
