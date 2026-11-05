import time
import random

N = 40

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin

    # --- Strategy: Star-Path Hybrid with Local Search for Chord Set S ---
    # Graph Structure:
    # Vertices: 0..39
    # Edges:
    #   1. Path edges: (i, i+1) for i=1..38. (38 edges)
    #   2. Spoke edges: (0, s) for s in S. (|S| edges)
    # Total edges = 38 + |S|.
    # Cycle of length L:
    #   We need a cycle of length L.
    #   A natural cycle in this structure is 0 -> u -> u+1 -> ... -> v -> 0.
    #   Length = 1 (edge 0-u) + (v-u) (path) + 1 (edge v-0) = v - u + 2.
    #   So we need v - u = L - 2.
    #   Let d = L - 2. We need d in [1, 38] (for L=3..40).
    #   We need a set S such that for every d in [1, 38], there exist u, v in S with v - u = d.
    #   Note: u and v must be in S. The path u..v uses vertices u, u+1, ..., v.
    #   These vertices are distinct and within 0..39.
    #   The cycle vertices are {0} U {u, ..., v}. Size is 1 + (v-u+1) = v-u+2 = L. Correct.
    #   Are the edges present?
    #   (0, u) is a spoke. (v, 0) is a spoke.
    #   (i, i+1) for i=u..v-1 are path edges.
    #   So yes, the cycle exists if such u, v exist.
    #
    # Goal: Minimize |S| such that the difference set of S covers [1, 38].
    # This is a "Golomb ruler" like problem but for covering all differences.
    # Actually, it's finding a subset of {0..39} (excluding 0 usually, or including?)
    # Wait, 0 is the center. S is a subset of {1..39}.
    # Differences are v - u where u, v in S.
    #
    # Let's search for S.
    # Minimum size?
    # If |S|=k, number of differences is k*(k-1)/2.
    # We need 38 differences.
    # k=9 -> 36 diffs. Not enough.
    # k=10 -> 45 diffs. Possible.
    # So minimum |S| is likely 10.
    # Total edges = 38 + 10 = 48.
    # Score = 48 - 40 = 8.
    # Current best score is 1.9474 (approx 59 edges? No, score is value. 1.9474 * 40? No.
    # The prompt says "Minimize edge count minus 40".
    # Current program has 40 (cycle) + 19 (chords) = 59 edges. Score = 19.
    # Wait, the prompt says "score 1.9474".
    # "The evaluator measures the raw value (lower is better). The search score is seed/value".
    # Seed value?
    # If seed is 19 edges extra, value=19. Score = 19/19 = 1.0?
    # "1.0 equals the seed program".
    # So if current program scores 1.9474, and it has 59 edges (19 extra), then:
    # 19 / value = 1.9474 => value = 19 / 1.9474 = 9.75?
    # Or maybe "value" is the metric.
    # "Minimize edge count minus 40".
    # If edges=59, value=19.
    # If seed has value 19, score=1.0.
    # If current scores 1.9474, then 19/value = 1.9474 => value = 9.75.
    # This implies the "current program" in the prompt description might be different from the code provided,
    # OR the "value" is not just edges-40.
    # Let's re-read: "Minimize edge count minus 40. This is an independently implemented n=40 construction search...".
    # "The evaluator measures the raw value (lower is better)."
    # "search score is seed/value".
    # If I produce 48 edges, value = 8.
    # If seed value is 19 (from 59 edges), then score = 19/8 = 2.375.
    # This would be an improvement over 1.9474.
    
    # Let's find a set S of size 10 that covers differences 1..38.
    
    target_diffs = set(range(1, 39))
    
    def get_coverage(S):
        S_sorted = sorted(S)
        diffs = set()
        for i in range(len(S_sorted)):
            for j in range(i+1, len(S_sorted)):
                d = S_sorted[j] - S_sorted[i]
                if d in target_diffs:
                    diffs.add(d)
        return diffs
        
    def is_valid(S):
        return get_coverage(S) == target_diffs

    best_S = None
    
    # Local Search
    # Initialize S with a known good spread or random
    # A good heuristic: spread elements evenly.
    # 0, 4, 8, 12, 16, 20, 24, 28, 32, 36?
    # Diffs: 4, 8, ... 36. Missing 1, 2, 3, 5...
    # We need small differences too.
    # Try: {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}? Diffs 1..9. Missing 10..38.
    # We need a mix.
    
    # Let's use a simulated annealing or simple hill climbing.
    # Start with random S of size 10.
    
    random.seed(42) # For reproducibility in search, though result should be deterministic if found
    
    for attempt in range(100):
        if time.time() > deadline - 5.0:
            break
            
        # Initialize S
        S = random.sample(range(1, 40), 10)
        
        current_cov = get_coverage(S)
        current_score = len(current_cov)
        
        for iter in range(2000):
            if time.time() > deadline - 5.0:
                break
                
            if current_score == len(target_diffs):
                best_S = S
                break
                
            # Pick a random index to change
            idx = random.randint(0, 9)
            old_val = S[idx]
            
            # Pick a new value
            new_val = random.randint(1, 39)
            
            if new_val in S and new_val != old_val:
                continue
                
            S[idx] = new_val
            
            new_cov = get_coverage(S)
            new_score = len(new_cov)
            
            if new_score >= current_score:
                current_score = new_score
                current_cov = new_cov
            else:
                # Revert
                S[idx] = old_val
        
        if best_S is not None:
            break
            
    if best_S is None:
        # Fallback to a known valid set if search fails
        # We need to find one. Let's try a few hardcoded candidates or a more systematic search.
        # Actually, let's just run a longer search if time permits, or use a fallback graph.
        # Fallback: The seed graph (59 edges).
        edges = set()
        for i in range(N):
            u, v = i, (i + 1) % N
            if u > v: u, v = v, u
            edges.add((u, v))
        for m in range(2, 21):
            edges.add((0, m))
            
        cycles = []
        for k in range(3, N + 1):
            if k == N:
                cycles.append(list(range(N)))
            elif k <= 21:
                cycles.append(list(range(k)))
            else:
                m = 41 - k
                cycle_verts = [0] + list(range(39, m - 1, -1))
                cycles.append(cycle_verts)
                
        return {'edges': list(edges), 'cycles': cycles}
        
    # Construct Graph
    edges = set()
    
    # Path edges: 1-2, ..., 38-39
    for i in range(1, 39):
        edges.add((i, i+1))
        
    # Spoke edges: 0-s
    for s in best_S:
        edges.add((0, s))
        
    # Note: We do NOT add edge (0,1) or (0,39) unless 1 or 39 is in S.
    # If 1 is not in S, there is no edge (0,1).
    # If 39 is not in S, there is no edge (0,39).
    # The path is 1-2-...-39.
    # So vertex 0 is only connected to S.
    
    # Construct Cycles
    cycles = []
    S_set = set(best_S)
    
    for L in range(3, N + 1):
        d = L - 2
        # Find u, v in S such that v - u = d
        found = False
        for u in best_S:
            v = u + d
            if v in S_set:
                # Cycle: 0 -> u -> ... -> v -> 0
                cycle_verts = [0] + list(range(u, v + 1))
                cycles.append(cycle_verts)
                found = True
                break
        if not found:
            # Should not happen if is_valid passed
            # But for safety, if it does, we have a problem.
            # Let's try the other direction? No, v-u=d is specific.
            # If coverage is correct, it must exist.
            raise Exception(f"Cycle not found for length {L}")
            
    return {'edges': list(edges), 'cycles': cycles}

if __name__ == "__main__":
    # Self-test
    res = construct()
    edges = set(tuple(sorted(e)) for e in res['edges'])
    valid = True
    for i, c in enumerate(res['cycles']):
        L = len(c)
        if L != i + 3:
            print(f"Cycle length mismatch at index {i}: expected {i+3}, got {L}")
            valid = False
        # Check distinct vertices
        if len(set(c)) != L:
            print(f"Cycle {L} has duplicate vertices")
            valid = False
        # Check edges
        for j in range(L):
            u, v = c[j], c[(j+1)%L]
            if (u, v) not in edges and (v, u) not in edges:
                print(f"Missing edge {u}-{v} for cycle {c}")
                valid = False
                break
    print(f"Edges: {len(res['edges'])}, Score Value: {len(res['edges'])-40}")
    if valid:
        print("All checks passed.")
    else:
        print("Validation failed.")
