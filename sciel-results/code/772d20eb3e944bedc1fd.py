"""Linear triple systems on 16 points with small independence number."""
import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 135  # safety margin before 150s kill
    
    random.seed(42)
    
    N = 16
    ALL_TRIPLES = list(itertools.combinations(range(N), 3))
    num_triples = len(ALL_TRIPLES)
    
    # Precompute triple bitmasks
    triple_masks = []
    for t in ALL_TRIPLES:
        mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        triple_masks.append(mask)
    
    # Precompute popcount for all 2^16 masks
    popcount = [0] * (1 << N)
    for i in range(1, 1 << N):
        popcount[i] = popcount[i >> 1] + (i & 1)
    
    # Precompute for each triple, which other triples conflict (share >= 2 points)
    # conflict_mask[i] is a bitset of triple indices that conflict with triple i
    conflict_mask = [0] * num_triples
    for i in range(num_triples):
        mi = triple_masks[i]
        for j in range(i + 1, num_triples):
            mj = triple_masks[j]
            shared = mi & mj
            if popcount[shared] >= 2:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)
    
    def independence_number(triples):
        """Find the largest subset of {0..15} containing no triple."""
        if not triples:
            return N
        
        # Build a set of triple masks for fast lookup
        tm_set = set()
        for t in triples:
            mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
            tm_set.add(mask)
        
        best = 0
        # Iterate masks from high popcount to low
        # Group by popcount for early termination
        for size in range(N, best, -1):
            if size <= best:
                break
            # Check all masks with this popcount
            found = False
            for mask in range(1 << N):
                if popcount[mask] != size:
                    continue
                # Check if any triple is a subset of mask
                has_triple = False
                for tm in tm_set:
                    if (mask & tm) == tm:
                        has_triple = True
                        break
                if not has_triple:
                    best = size
                    found = True
                    break
            if found:
                break
        return best
    
    def find_max_independent_set(triples):
        """Find an actual maximum independent set (returns the mask)."""
        if not triples:
            return (1 << N) - 1
        
        tm_set = set()
        for t in triples:
            mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
            tm_set.add(mask)
        
        best_mask = 0
        best_size = 0
        for size in range(N, best_size, -1):
            found = False
            for mask in range(1 << N):
                if popcount[mask] != size:
                    continue
                has_triple = False
                for tm in tm_set:
                    if (mask & tm) == tm:
                        has_triple = True
                        break
                if not has_triple:
                    best_mask = mask
                    best_size = size
                    found = True
                    break
            if found:
                break
        return best_mask
    
    def is_linear(triples):
        """Check if any two triples share at most one point."""
        masks = [(1 << t[0]) | (1 << t[1]) | (1 << t[2]) for t in triples]
        for i in range(len(masks)):
            for j in range(i + 1, len(masks)):
                if popcount[masks[i] & masks[j]] > 1:
                    return False
        return True
    
    def random_greedy_packing(seed_offset=0):
        """Build a dense linear triple system via randomized greedy packing."""
        rng = random.Random(42 + seed_offset)
        used_pairs = set()
        triples = []
        
        candidates = list(range(num_triples))
        rng.shuffle(candidates)
        
        for idx in candidates:
            t = ALL_TRIPLES[idx]
            pairs = [
                frozenset((t[0], t[1])),
                frozenset((t[0], t[2])),
                frozenset((t[1], t[2]))
            ]
            can_add = True
            for p in pairs:
                if p in used_pairs:
                    can_add = False
                    break
            if can_add:
                for p in pairs:
                    used_pairs.add(p)
                triples.append(t)
        
        return triples
    
    def can_add_triple(triples, new_triple):
        """Check if new_triple can be added without violating linearity."""
        nm = (1 << new_triple[0]) | (1 << new_triple[1]) | (1 << new_triple[2])
        for t in triples:
            tm = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
            if popcount[nm & tm] > 1:
                return False
        return True
    
    def local_search(triples, max_time=100):
        """Improve the system by replacing/adding triples to reduce independence number."""
        best_triples = triples[:]
        best_score = independence_number(best_triples)
        
        current_triples = triples[:]
        current_score = best_score
        
        timeout = time.time() + max_time
        no_improve_count = 0
        
        while time.time() < timeout:
            if no_improve_count > 2000:
                break
            
            # Find the current max independent set
            iset_mask = find_max_independent_set(current_triples)
            iset_points = [i for i in range(N) if (iset_mask >> i) & 1]
            iset_size = len(iset_points)
            
            if iset_size <= 3:
                break
            
            # Strategy 1: Add a triple that is a subset of the independent set
            added = False
            if len(iset_points) >= 3:
                candidates = list(itertools.combinations(iset_points, 3))
                random.shuffle(candidates)
                for new_triple in candidates:
                    if can_add_triple(current_triples, new_triple):
                        current_triples.append(new_triple)
                        new_score = independence_number(current_triples)
                        if new_score < current_score:
                            current_score = new_score
                            if new_score < best_score:
                                best_score = new_score
                                best_triples = current_triples[:]
                                no_improve_count = 0
                            added = True
                            break
                        else:
                            current_triples.pop()
            
            # Strategy 2: Replace a random triple with one from the independent set
            if not added and len(current_triples) > 0:
                idx = random.randint(0, len(current_triples) - 1)
                old_triple = current_triples[idx]
                
                # Try replacing with a triple from the independent set
                candidates = list(itertools.combinations(iset_points, 3))
                random.shuffle(candidates)
                
                for new_triple in candidates:
                    # Check if new_triple conflicts with all other triples (excluding idx)
                    valid = True
                    nm = (1 << new_triple[0]) | (1 << new_triple[1]) | (1 << new_triple[2])
                    for j, t in enumerate(current_triples):
                        if j == idx:
                            continue
                        tm = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
                        if popcount[nm & tm] > 1:
                            valid = False
                            break
                    
                    if not valid:
                        continue
                    
                    current_triples[idx] = new_triple
                    new_score = independence_number(current_triples)
                    if new_score < current_score:
                        current_score = new_score
                        if new_score < best_score:
                            best_score = new_score
                            best_triples = current_triples[:]
                            no_improve_count = 0
                        added = True
                        break
                    else:
                        current_triples[idx] = old_triple
                
                if not added:
                    # Try random replacement
                    for attempt in range(5):
                        new_triple = tuple(sorted(random.sample(range(N), 3)))
                        nm = (1 << new_triple[0]) | (1 << new_triple[1]) | (1 << new_triple[2])
                        valid = True
                        for j, t in enumerate(current_triples):
                            if j == idx:
                                continue
                            tm = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
                            if popcount[nm & tm] > 1:
                                valid = False
                                break
                        
                        if not valid:
                            continue
                        
                        current_triples[idx] = new_triple
                        new_score = independence_number(current_triples)
                        if new_score < current_score:
                            current_score = new_score
                            if new_score < best_score:
                                best_score = new_score
                                best_triples = current_triples[:]
                                no_improve_count = 0
                            added = True
                            break
                        else:
                            current_triples[idx] = old_triple
            
            if not added:
                no_improve_count += 1
                # Occasionally try adding a random triple
                if len(current_triples) < 80 and random.random() < 0.2:
                    new_triple = tuple(sorted(random.sample(range(N), 3)))
                    if can_add_triple(current_triples, new_triple):
                        current_triples.append(new_triple)
                        new_score = independence_number(current_triples)
                        if new_score < current_score:
                            current_score = new_score
                            if new_score < best_score:
                                best_score = new_score
                                best_triples = current_triples[:]
                                no_improve_count = 0
                        else:
                            current_triples.pop()
        
        return best_triples, best_score
    
    # Phase 1: Generate initial systems with different seeds
    best_triples = None
    best_score = N + 1
    
    for seed_offset in range(20):
        if time.time() > deadline - 30:
            break
        initial = random_greedy_packing(seed_offset)
        if not initial:
            continue
        score = independence_number(initial)
        if score < best_score:
            best_score = score
            best_triples = initial[:]
    
    if best_triples is None:
        best_triples = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14]]
        best_score = independence_number(best_triples)
    
    # Phase 2: Local search
    remaining_time = deadline - time.time() - 5
    if remaining_time > 10:
        best_triples, best_score = local_search(best_triples, max_time=remaining_time)
    
    # Final check
    if not is_linear(best_triples):
        return [[3*i, 3*i+1, 3*i+2] for i in range(5)]
    
    return [list(t) for t in best_triples]
