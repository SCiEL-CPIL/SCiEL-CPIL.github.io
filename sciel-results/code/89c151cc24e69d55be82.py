"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def _precompute_triples(N=180):
    """Precompute all forbidden triples (a, b, c) with a < b < c, distinct,
    satisfying 1/a = 1/b + 1/c."""
    triples = []
    for a in range(1, N + 1):
        a2 = a * a
        for d in range(1, a):
            if a2 % d == 0:
                e = a2 // d
                b = a + d
                c = a + e
                if b < c and c <= N and b > a:
                    triples.append((a, b, c))
    return triples

def _check_valid(S, triples):
    """Check if set S has no forbidden triple."""
    S_set = set(S)
    for (a, b, c) in triples:
        if a in S_set and b in S_set and c in S_set:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    N = 180
    triples = _precompute_triples(N)
    Ntriples = len(triples)
    
    # Build element-to-triples mapping
    elem_triples = defaultdict(list)
    for idx, (a, b, c) in enumerate(triples):
        elem_triples[a].append(idx)
        elem_triples[b].append(idx)
        elem_triples[c].append(idx)
    
    best_S = set()
    best_size = 0
    
    def greedy_removal(seed=0):
        """Start with all elements, greedily remove to break triples."""
        random.seed(seed)
        S = set(range(1, N + 1))
        
        # Precompute conflict counts
        def get_conflict_counts(S_set):
            counts = defaultdict(int)
            for (a, b, c) in triples:
                if a in S_set and b in S_set and c in S_set:
                    counts[a] += 1
                    counts[b] += 1
                    counts[c] += 1
            return counts
        
        changed = True
        while changed:
            changed = False
            if time.time() > deadline - 5.0:
                break
            
            counts = get_conflict_counts(S)
            if not counts:
                break
            
            max_conf = max(counts.values())
            candidates = [x for x in S if counts.get(x, 0) == max_conf]
            x_to_remove = random.choice(candidates)
            S.remove(x_to_remove)
            changed = True
        
        return S
    
    def local_search(S, max_iters=5000):
        """Local search: try to add elements, swap elements."""
        S = set(S)
        current_size = len(S)
        improved = True
        iters = 0
        
        while improved and iters < max_iters and time.time() < deadline - 3.0:
            improved = False
            iters += 1
            
            # Try adding each element not in S
            candidates = [x for x in range(1, N + 1) if x not in S]
            random.shuffle(candidates)
            
            for x in candidates:
                if time.time() > deadline - 3.0:
                    break
                
                # Check if x can be added
                can_add = True
                for idx in elem_triples.get(x, []):
                    a, b, c = triples[idx]
                    others = [y for y in (a, b, c) if y != x]
                    if all(o in S for o in others):
                        can_add = False
                        break
                
                if can_add:
                    S.add(x)
                    current_size += 1
                    improved = True
                    if current_size > best_size:
                        best_S = set(S)
                        best_size = current_size
                    break  # Restart loop after adding
            
            # Try swaps: remove one, add two
            if not improved and iters < max_iters // 2:
                # Try removing an element with high conflict and adding others
                S_list = list(S)
                random.shuffle(S_list)
                
                for rem in S_list[:20]:  # Try a few removals
                    if time.time() > deadline - 3.0:
                        break
                    
                    S_test = S - {rem}
                    added = 0
                    
                    for x in range(1, N + 1):
                        if x in S_test or x == rem:
                            continue
                        
                        can_add = True
                        for idx in elem_triples.get(x, []):
                            a, b, c = triples[idx]
                            others = [y for y in (a, b, c) if y != x]
                            if all(o in S_test for o in others):
                                can_add = False
                                break
                        
                        if can_add:
                            S_test.add(x)
                            added += 1
                            if added >= 2:
                                break
                    
                    if added > 1:  # Net gain of at least 1
                        S = S_test
                        current_size = len(S)
                        improved = True
                        if current_size > best_size:
                            best_S = set(S)
                            best_size = current_size
                        break
        
        return S
    
    # Phase 1: Greedy removal with multiple seeds
    for seed in range(30):
        if time.time() > deadline - 15.0:
            break
        
        S = greedy_removal(seed)
        if len(S) > best_size:
            best_S = set(S)
            best_size = len(S)
        
        # Local search on this solution
        S = local_search(S, max_iters=2000)
        if len(S) > best_size:
            best_S = set(S)
            best_size = len(S)
    
    # Phase 2: Start from large suffix and try to add
    S = set(range(91, N + 1))
    for x in range(1, 91):
        if time.time() > deadline - 10.0:
            break
        can_add = True
        for idx in elem_triples.get(x, []):
            a, b, c = triples[idx]
            others = [y for y in (a, b, c) if y != x]
            if all(o in S for o in others):
                can_add = False
                break
        if can_add:
            S.add(x)
    
    if len(S) > best_size:
        best_S = set(S)
        best_size = len(S)
    
    S = local_search(S, max_iters=3000)
    if len(S) > best_size:
        best_S = set(S)
        best_size = len(S)
    
    # Phase 3: Randomized restarts with biased initialization
    for trial in range(20):
        if time.time() > deadline - 5.0:
            break
        
        # Random initial set with bias toward larger elements
        S = set()
        for x in range(1, N + 1):
            prob = 0.3 + 0.5 * (x / N)  # Higher probability for larger x
            if random.random() < prob:
                S.add(x)
        
        # Remove conflicts
        changed = True
        while changed:
            changed = False
            for (a, b, c) in triples:
                if a in S and b in S and c in S:
                    # Remove the one with most conflicts
                    counts = [len(elem_triples.get(x, [])) for x in (a, b, c)]
                    max_idx = counts.index(max(counts))
                    S.remove((a, b, c)[max_idx])
                    changed = True
                    break
        
        if len(S) > best_size:
            best_S = set(S)
            best_size = len(S)
        
        S = local_search(S, max_iters=1500)
        if len(S) > best_size:
            best_S = set(S)
            best_size = len(S)
    
    # Final validation
    result = sorted(best_S)
    assert len(result) > 0
    assert _check_valid(result, triples), "Result is not valid!"
    
    return result
# EVOLVE-BLOCK-END
