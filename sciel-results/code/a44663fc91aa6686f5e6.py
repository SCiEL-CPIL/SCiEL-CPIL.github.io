import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    n = 256
    
    # Build conflict graph: edge (i, j) exists if i+j is a perfect square (i <= j)
    # Also include self-loop if 2*i is a perfect square (i = j case)
    # For the independent set problem, we need to avoid both:
    # - a != b: a+b is a perfect square
    # - a == b: 2a is a perfect square
    
    # Determine which values have 2a as a perfect square (these can't be in the set)
    bad_self = set()
    for a in range(1, n + 1):
        s = 2 * a
        r = int(math.isqrt(s))
        if r * r == s:
            bad_self.add(a)
    
    # Build adjacency: for each pair (a, b) with a < b, check if a+b is a perfect square
    # Use a bitset or list of sets for efficiency
    # 256 nodes, so at most 256*255/2 = 32640 edges - manageable
    
    adj = [set() for _ in range(n + 1)]  # 1-indexed
    for a in range(1, n + 1):
        if a in bad_self:
            continue
        for b in range(a + 1, n + 1):
            if b in bad_self:
                continue
            s = a + b
            r = int(math.isqrt(s))
            if r * r == s:
                adj[a].add(b)
                adj[b].add(a)
    
    # Filter candidates: exclude those with bad_self
    candidates = [a for a in range(1, n + 1) if a not in bad_self]
    cand_idx = {a: i for i, a in enumerate(candidates)}
    m = len(candidates)
    
    # Build adjacency in candidate-index space
    cand_adj = [set() for _ in range(m)]
    for a in candidates:
        ia = cand_idx[a]
        for b in adj[a]:
            ib = cand_idx[b]
            cand_adj[ia].add(ib)
    
    # Greedy initialization: sort by degree (ascending) and add if no conflict
    def greedy_init():
        # Try multiple random orderings and greedy picks
        best_set = set()
        for _ in range(50):
            random.shuffle(candidates)
            # Sort by degree ascending within the shuffled list for better packing
            order = sorted(candidates, key=lambda x: (len(cand_adj[cand_idx[x]]), random.random()))
            current = set()
            for a in order:
                ia = cand_idx[a]
                conflict = False
                for b in current:
                    if b in adj[a]:
                        conflict = True
                        break
                if not conflict:
                    current.add(a)
            if len(current) > len(best_set):
                best_set = current
        return best_set
    
    best_set = greedy_init()
    best_size = len(best_set)
    
    # Local search: try adding, removing, swapping to improve
    def is_valid_add(a, current_set):
        for b in current_set:
            if b in adj[a]:
                return False
        # Check self: already filtered
        return True
    
    def local_search(current_set):
        current = set(current_set)
        improved = True
        while improved:
            improved = False
            # Try adding each non-member
            for a in candidates:
                if a in current:
                    continue
                if is_valid_add(a, current):
                    current.add(a)
                    improved = True
                    break
            # Try swaps: remove one, add two
            if not improved:
                for a in list(current):
                    for b in candidates:
                        if b in current:
                            continue
                        if b in adj[a]:
                            continue
                        # Try removing a, adding b
                        trial = current - {a}
                        if is_valid_add(b, trial):
                            if len(trial) + 1 > len(current):
                                current = trial | {b}
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    continue
                # Try remove one, add two
                for a in list(current):
                    removed = current - {a}
                    added_count = 0
                    added = []
                    for b in candidates:
                        if b in current or b == a:
                            continue
                        if b in adj[a]:  # not directly related, but check against remaining
                            pass
                        conflict = False
                        for c in removed:
                            if b in adj[c]:
                                conflict = True
                                break
                        if not conflict and is_valid_add(b, removed):
                            added.append(b)
                            removed = removed | {b}
                            added_count += 1
                        if added_count >= 2:
                            break
                    if added_count >= 2:
                        current = removed
                        improved = True
                        break
        return current
    
    # Run local search with time budget
    while time.time() < deadline - 5.0:
        candidate_set = local_search(best_set)
        if len(candidate_set) > best_size:
            best_size = len(candidate_set)
            best_set = candidate_set
        else:
            # Try perturbation: random restart with some fixed elements
            fixed = set(random.sample(list(best_set), min(len(best_set), max(1, len(best_set) // 2))))
            perturbed = set(fixed)
            for a in candidates:
                if a in perturbed:
                    continue
                if is_valid_add(a, perturbed):
                    perturbed.add(a)
            candidate_set = local_search(perturbed)
            if len(candidate_set) > best_size:
                best_size = len(candidate_set)
                best_set = candidate_set
    
    # Final polish: try to add any remaining
    for a in candidates:
        if a not in best_set and is_valid_add(a, best_set):
            best_set.add(a)
    
    return list(best_set)
