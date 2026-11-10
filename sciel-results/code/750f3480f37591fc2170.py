# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    deadline = time.time() + 620  # safety margin before 630s kill

    # Precompute forbidden set
    forbidden = set()
    for x in range(1, m):
        v = (x * x) % m
        if v != 0:
            forbidden.add(v)

    # Precompute conflict matrix: conflict[i][j] = True if (i-j)%m or (j-i)%m is in forbidden
    # Since forbidden is symmetric (if d in F then -d mod m may or may not be in F),
    # actually we need: conflict if (i-j) mod m in F OR (j-i) mod m in F
    # But note: if d in F, is -d mod m in F? Let's just check both.
    # Actually the condition is: neither (a-b) mod m nor (b-a) mod m in F
    # So conflict exists if (a-b)%m in F OR (b-a)%m in F
    # Note (b-a)%m = m - (a-b)%m if a!=b, so we need both d and m-d checked.
    
    # Build adjacency: for each vertex, which other vertices conflict with it
    # conflict[i] = set of j such that i and j cannot both be in solution
    conflict = [set() for _ in range(m)]
    for i in range(m):
        for j in range(i+1, m):
            d1 = (i - j) % m
            d2 = (j - i) % m
            if d1 in forbidden or d2 in forbidden:
                conflict[i].add(j)
                conflict[j].add(i)

    # Convert to numpy boolean matrix for fast lookup
    # conflict_mat[i,j] = True if i and j conflict
    conflict_mat = np.zeros((m, m), dtype=bool)
    for i in range(m):
        for j in conflict[i]:
            conflict_mat[i, j] = True
            conflict_mat[j, i] = True

    def is_valid_subset(subset):
        """Check if subset is independent (no conflicts)."""
        s = set(subset)
        for a in s:
            for b in conflict[a]:
                if b in s:
                    return False
        return True

    def greedy_init(rng):
        """Greedy: add elements in random order if they don't conflict with already selected."""
        order = list(range(m))
        rng.shuffle(order)
        selected = set()
        for v in order:
            # Check if v conflicts with any already selected
            ok = True
            for s in selected:
                if conflict_mat[v, s]:
                    ok = False
                    break
            if ok:
                selected.add(v)
        return selected

    def local_search(selected, rng, deadline):
        """
        Local search: try to improve by:
        1. Adding any vertex that doesn't conflict
        2. Swapping: remove one, add one or two
        """
        selected = set(selected)
        best_size = len(selected)
        no_improve_count = 0
        
        while time.time() < deadline - 5:
            improved = False
            
            # Try adding a vertex
            for v in range(m):
                if v in selected:
                    continue
                # Check if v conflicts with any selected
                ok = True
                for s in selected:
                    if conflict_mat[v, s]:
                        ok = False
                        break
                if ok:
                    selected.add(v)
                    improved = True
                    break
            
            if not improved:
                # Try 1-for-1 swap: remove a, add b
                for a in list(selected):
                    for b in range(m):
                        if b in selected:
                            continue
                        # Check if b conflicts with (selected - {a})
                        ok = True
                        for s in selected:
                            if s == a:
                                continue
                            if conflict_mat[b, s]:
                                ok = False
                                break
                        if ok:
                            selected.discard(a)
                            selected.add(b)
                            improved = True
                            break
                    if improved:
                        break
            
            if not improved:
                # Try 1-for-2 swap: remove a, add b and c
                for a in list(selected):
                    found = False
                    for b in range(m):
                        if b in selected:
                            continue
                        # Check b against selected - {a}
                        ok_b = True
                        for s in selected:
                            if s == a:
                                continue
                            if conflict_mat[b, s]:
                                ok_b = False
                                break
                        if not ok_b:
                            continue
                        for c in range(m):
                            if c in selected or c == b:
                                continue
                            # Check c against selected - {a} and b
                            ok_c = True
                            for s in selected:
                                if s == a:
                                    continue
                                if conflict_mat[c, s]:
                                    ok_c = False
                                    break
                            if ok_c and conflict_mat[b, c]:
                                ok_c = False
                            if ok_c:
                                selected.discard(a)
                                selected.add(b)
                                selected.add(c)
                                improved = True
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break
            
            if improved:
                no_improve_count = 0
                if len(selected) > best_size:
                    best_size = len(selected)
            else:
                no_improve_count += 1
                # Perturb: remove a random element to escape local optimum
                if no_improve_count >= 20:
                    if selected:
                        to_remove = rng.sample(list(selected), min(3, len(selected)))
                        for r in to_remove:
                            selected.discard(r)
                    no_improve_count = 0
    
    return selected

    # Multi-start local search
    best_subset = set()
    rng = random.Random(42)
    
    # First, try a few greedy inits
    for trial in range(20):
        if time.time() > deadline - 10:
            break
        sel = greedy_init(rng)
        if len(sel) > len(best_subset):
            best_subset = set(sel)
        # Do local search on this
        sel = local_search(sel, rng, deadline)
        if len(sel) > len(best_subset):
            best_subset = set(sel)
    
    # Continue local search on best found
    if time.time() < deadline - 5:
        best_subset = local_search(best_subset, rng, deadline)
    
    return np.array(sorted(best_subset), dtype=np.int64)

# EVOLVE-BLOCK-END
