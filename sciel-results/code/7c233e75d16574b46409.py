import random
import time

N = 40

def construct():
    start_time = time.time()
    timeout = 5.0  # 5 seconds for optimization, plenty of margin
    
    # Structure: Path 1-2-...-39 and Spokes 0-s for s in S
    # Path edges: (i, i+1) for i in 1..38
    # Spoke edges: (0, s) for s in S
    # Total edges: 38 + len(S)
    # Value: 38 + len(S) - 40 = len(S) - 2
    
    # We need cycles of length L=3..40.
    # Cycle for length L uses chord (0, u) and (0, v) ? No.
    # Cycle: 0 -> u -> u+1 -> ... -> v -> 0.
    # Edges: (0,u), (u, u+1), ..., (v-1, v), (v, 0).
    # This requires (0,u) and (0,v) to be in the graph?
    # No, the path is on the backbone.
    # The cycle is 0-u-(u+1)-...-v-0.
    # This requires edges (0,u) and (0,v) to exist?
    # Yes, the first edge is (0,u) and the last is (v,0).
    # So u and v must be in S.
    # The length is (v-u) + 2.
    # We need to cover lengths 3..40, so differences d = L-2 must cover 1..38.
    # So we need a set S such that {v-u | u,v in S, v>u} covers {1, ..., 38}.
    
    target_diffs = set(range(1, 39))
    
    def get_coverage(S):
        S_sorted = sorted(S)
        diffs = set()
        for i in range(len(S_sorted)):
            for j in range(i+1, len(S_sorted)):
                d = S_sorted[j] - S_sorted[i]
                if d in target_diffs:
                    diffs.add(d)
        return len(diffs)

    def is_valid(S):
        return get_coverage(S) == len(target_diffs)

    best_S = None
    best_coverage = 0
    
    # Try to find a small set S
    # Theoretical min size for basis of 1..38 is around 10-12.
    # We will try sizes 10, 11, 12.
    
    # Heuristic Initialization: Spread out
    # A known good set for similar problems often involves powers of 2 or similar sparse sets.
    # Let's use a greedy initialization.
    
    for size in [10, 11, 12]:
        if time.time() - start_time > timeout:
            break
            
        # Initialize S with some random spread
        S = set()
        # Always include 1 and 39 to cover extreme diffs? 
        # Diff 38 requires pair (1, 39) or (0, 38) etc.
        # Since S is subset of 1..39, diff 38 needs (1, 39).
        # So 1 and 39 must be in S.
        S.add(1)
        S.add(39)
        
        # Fill rest randomly
        candidates = list(range(2, 39))
        random.shuffle(candidates)
        S.update(candidates[:size-2])
        
        current_score = get_coverage(S)
        best_coverage = max(best_coverage, current_score)
        if current_score == len(target_diffs):
            best_S = list(S)
            break
            
        # Local Search
        no_improve = 0
        max_iters = 5000
        for iter in range(max_iters):
            if time.time() - start_time > timeout:
                break
            if current_score == len(target_diffs):
                best_S = list(S)
                break
                
            # Pick a random element to change
            idx = random.randint(0, len(S)-1)
            # We need to pick an index from the list, but S is a set.
            S_list = list(S)
            old_val = S_list[idx]
            
            # Pick a new value
            new_val = random.randint(1, 39)
            
            # Ensure distinctness
            if new_val in S and new_val != old_val:
                continue
                
            # Try change
            S.remove(old_val)
            S.add(new_val)
            
            new_score = get_coverage(S)
            
            if new_score > current_score:
                current_score = new_score
                no_improve = 0
                if current_score > best_coverage:
                    best_coverage = current_score
                    if current_score == len(target_diffs):
                        best_S = list(S)
                        break
            else:
                # Revert
                S.remove(new_val)
                S.add(old_val)
                no_improve += 1
                if no_improve > 100:
                    # Restart with new random init
                    S = {1, 39}
                    candidates = list(range(2, 39))
                    random.shuffle(candidates)
                    S.update(candidates[:size-2])
                    current_score = get_coverage(S)
                    no_improve = 0
                    break
        
        if best_S is not None:
            break

    # Fallback to Ladder if search failed or didn't find a good set
    # Ladder Score is 19 (Value 19).
    # If we found S of size 10, Value is 8. Score ~4.6.
    # If we found S of size 11, Value is 9. Score ~4.1.
    # If we found nothing, we use Ladder.
    
    if best_S is None:
        # Check if best_coverage was high enough to be better than Ladder?
        # Ladder Value = 19.
        # If we have a set S with size K, Value = K-2.
        # We want K-2 < 19 => K < 21.
        # If we didn't find a *valid* set, we can't use it.
        # So we must use the Ladder fallback.
        
        # Ladder Construction
        edges = []
        # Top path 0..18
        for i in range(19):
            edges.append((i, i+1))
        # Bottom path 20..39
        for i in range(19):
            edges.append((i+20, i+21))
        # Rungs
        for i in range(20):
            edges.append((i, i+20))
        # Chord (0, 21)
        edges.append((0, 21))
        
        cycles = []
        for k in range(3, 41):
            if k % 2 == 1:
                m = (k - 1) // 2
                cyc = list(range(0, m+1))
                cyc.extend(range(m+20, 20, -1))
                cycles.append(cyc)
            else:
                m = k // 2 - 1
                cyc = list(range(0, m+1))
                cyc.extend(range(m+20, 19, -1))
                cycles.append(cyc)
        
        return {'edges': edges, 'cycles': cycles}

    # Construct graph with best_S
    path_edges = [(i, i+1) for i in range(1, N-1)]
    spoke_edges = [(0, s) for s in best_S]
    
    edges = path_edges + spoke_edges
    
    # Build mapping from diff to pair (u, v)
    S_set = set(best_S)
    cycles = []
    for L in range(3, N+1):
        diff_needed = L - 2
        # Find u, v in best_S such that v - u == diff_needed
        found = False
        for u in best_S:
            v = u + diff_needed
            if v in S_set:
                # Cycle: 0 -> u -> ... -> v -> 0
                cycle_verts = [0] + list(range(u, v+1))
                cycles.append(cycle_verts)
                found = True
                break
        if not found:
            # Should not happen if is_valid passed
            raise Exception("Cycle not found for length " + str(L))
            
    return {'edges': edges, 'cycles': cycles}
