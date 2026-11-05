import math
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    
    # 255 = 3 * 5 * 17
    # Build adjacency: gcd(a,b) > 1
    n = 255
    vertices = list(range(1, n + 1))
    
    # Precompute gcd matrix (only for numbers that share factor with 255)
    # A number must share a factor with 255 to be in any valid set with 255
    def shares_factor_with_255(x):
        return math.gcd(x, 255) > 1
    
    candidates = [x for x in range(1, n + 1) if shares_factor_with_255(x)]
    candidates_set = set(candidates)
    
    # Precompute gcd for all pairs of candidates
    # This is at most 135 candidates (multiples of 3, 5, or 17 up to 255)
    # Let's count: multiples of 3: 85, multiples of 5: 51, multiples of 17: 15
    # Inclusion-exclusion: 85+51+15 - gcd(3,5)=15 - gcd(3,17)=3 - gcd(5,17)=5 + gcd(3,5,17)=1
    # = 85+51+15-15-3-5+1 = 129
    cand_idx = {v: i for i, v in enumerate(candidates)}
    m = len(candidates)
    
    # Build adjacency as bitsets for speed
    adj = [0] * m
    for i in range(m):
        vi = candidates[i]
        for j in range(i + 1, m):
            vj = candidates[j]
            if math.gcd(vi, vj) > 1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # 255 must be in the set
    idx_255 = cand_idx[255]
    
    # Greedy maximum clique starting with 255
    # At each step, add the vertex with the most connections to current clique
    def greedy_clique(start_idx, adj, m, deadline):
        clique = {start_idx}
        # Candidates that connect to all in clique
        # Start: must connect to 255
        available = [i for i in range(m) if i != start_idx and (adj[start_idx] >> i) & 1]
        
        while available:
            if time.time() > deadline:
                break
            # Find vertex with max connections to current clique
            best = -1
            best_score = -1
            for i in available:
                score = 0
                for c in clique:
                    if (adj[i] >> c) & 1:
                        score += 1
                if score > best_score:
                    best_score = score
                    best = i
            if best == -1:
                break
            clique.add(best)
            # Update available: must connect to all in clique
            new_available = []
            for i in available:
                if i == best:
                    continue
                ok = True
                for c in clique:
                    if not ((adj[i] >> c) & 1):
                        ok = False
                        break
                if ok:
                    new_available.append(i)
            available = new_available
        
        return clique
    
    # Try greedy from 255
    best_clique = greedy_clique(idx_255, adj, m, deadline)
    
    # Also try the simple multiples-of-3 baseline
    multiples_3 = [x for x in range(3, 256, 3)]
    base_size = len(multiples_3)  # 85
    
    greedy_size = len(best_clique)
    
    # Try multiple greedy restarts with different orderings
    if time.time() < deadline:
        # Try removing some elements from multiples of 3 and adding others
        # Alternative: try greedy with different tie-breaking
        import random
        random.seed(42)
        
        best_size = max(greedy_size, base_size)
        best_set = multiples_3 if base_size >= greedy_size else [candidates[i] for i in best_clique]
        
        # Try several random orderings for greedy
        for trial in range(50):
            if time.time() > deadline:
                break
            # Random perturbation: start with 255, then pick from available randomly among top-k
            clique = {idx_255}
            available = [i for i in range(m) if i != idx_255 and (adj[idx_255] >> i) & 1]
            
            while available:
                if time.time() > deadline:
                    break
                # Score all available
                scored = []
                for i in available:
                    score = sum(1 for c in clique if (adj[i] >> c) & 1)
                    scored.append((score, i))
                scored.sort(reverse=True)
                # Pick from top min(5, len) randomly
                k = min(5, len(scored))
                chosen = random.choice(scored[:k])[1]
                clique.add(chosen)
                new_available = []
                for i in available:
                    if i == chosen:
                        continue
                    ok = True
                    for c in clique:
                        if not ((adj[i] >> c) & 1):
                            ok = False
                            break
                    if ok:
                        new_available.append(i)
                available = new_available
            
            if len(clique) > best_size:
                best_size = len(clique)
                best_set = [candidates[i] for i in clique]
    
    # Ensure 255 is in the result
    if 255 not in best_set:
        best_set.append(255)
    
    return best_set
