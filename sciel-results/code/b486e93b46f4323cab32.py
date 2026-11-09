import time
import random
import sys

N_POINTS = 16
DEADLINE = 140.0

def construct():
    start_time = time.time()
    random.seed(42)
    
    # Precompute all 560 triples
    all_triples = []
    for i in range(N_POINTS):
        for j in range(i+1, N_POINTS):
            for k in range(j+1, N_POINTS):
                all_triples.append((i, j, k))
    
    # Precompute masks
    all_masks = [(1<<i)|(1<<j)|(1<<k) for (i,j,k) in all_triples]
    
    # Precompute pair->triples index for fast compatibility
    # pair is a frozenset of two points
    pair_to_triples = {}
    for idx, (i,j,k) in enumerate(all_triples):
        p1 = (i,j) if i<j else (j,i)
        p2 = (i,k) if i<k else (k,i)
        p3 = (j,k) if j<k else (k,j)
        for p in [p1, p2, p3]:
            if p not in pair_to_triples:
                pair_to_triples[p] = []
            pair_to_triples[p].append(idx)
    
    def check_linearity(system):
        """Check if system is a valid linear triple system."""
        used_pairs = set()
        for t in system:
            pairs = [(t[0],t[1]), (t[0],t[2]), (t[1],t[2])]
            for p in pairs:
                if p in used_pairs:
                    return False
                used_pairs.add(p)
        return True
    
    def exact_alpha(masks):
        """Compute exact independence number using branch and bound on 16 points."""
        # Independent set = set S with no mask fully contained in S
        # We want max |S| such that for all m in masks: m is NOT a subset of S
        # Equivalently: for each m, at least one point of m is NOT in S.
        
        # Branch and bound: try to build a large independent set.
        # State: current set S (as bitmask), remaining points to consider
        # Pruning: if |S| + remaining_count <= best_found, prune.
        
        best = [0]
        n = N_POINTS
        
        def branch(S_mask, remaining_mask, depth):
            # Upper bound: |S| + popcount(remaining)
            ub = bin(S_mask).count('1') + bin(remaining_mask).count('1')
            if ub <= best[0]:
                return
            
            if remaining_mask == 0:
                if bin(S_mask).count('1') > best[0]:
                    best[0] = bin(S_mask).count('1')
                return
            
            # Try excluding a point (to keep set independent)
            # Or include a point if it doesn't violate
            # We iterate over points in remaining_mask
            
            # Optimization: pick a point, try including it first (to find large sets quickly)
            # Then try excluding it.
            
            # Find a violating triple to guide branching
            # If S_mask already contains a full triple, this branch is invalid.
            # Actually, we build S incrementally, so we check when adding.
            
            # Get list of points in remaining
            pts = []
            rm = remaining_mask
            while rm:
                lsb = rm & (-rm)
                p = lsb.bit_length() - 1
                pts.append(p)
                rm ^= lsb
            
            for p in pts:
                # Try including p
                new_S = S_mask | (1 << p)
                # Check if any triple is fully contained in new_S
                valid = True
                for m in masks:
                    if (m & new_S) == m:
                        valid = False
                        break
                if valid:
                    new_rem = remaining_mask & ~(1 << p)
                    branch(new_S, new_rem, depth+1)
                
                # Try excluding p
                new_rem = remaining_mask & ~(1 << p)
                branch(S_mask, new_rem, depth+1)
                
                # Time check
                if time.time() - start_time > DEADLINE - 5:
                    return
        
        branch(0, (1 << N_POINTS) - 1, 0)
        return best[0]
    
    def fast_alpha(masks):
        """Fast heuristic for alpha: greedy removal."""
        S_mask = (1 << N_POINTS) - 1
        # Precompute point degrees
        degrees = [0]*N_POINTS
        for m in masks:
            for p in range(N_POINTS):
                if (m >> p) & 1:
                    degrees[p] += 1
        
        changed = True
        while changed:
            changed = False
            # Find violating triple
            viol = -1
            for i, m in enumerate(masks):
                if (m & S_mask) == m:
                    viol = i
                    break
            if viol == -1:
                break
            m = masks[viol]
            # Remove point with highest degree among those in m and S
            best_p = -1
            best_deg = -1
            for p in range(N_POINTS):
                if (m >> p) & 1 and (S_mask >> p) & 1:
                    if degrees[p] > best_deg:
                        best_deg = degrees[p]
                        best_p = p
            if best_p >= 0:
                S_mask &= ~(1 << best_p)
                # Update degrees? For speed, skip.
                changed = True
        return bin(S_mask).count('1')
    
    # Phase 1: Generate diverse initial systems using different random seeds
    # and different greedy strategies.
    candidates = []
    
    for seed in range(20):
        random.seed(seed)
        shuffled = all_triples[:]
        random.shuffle(shuffled)
        selected = []
        used_pairs = set()
        for t in shuffled:
            pairs = [(t[0],t[1]), (t[0],t[2]), (t[1],t[2])]
            if all(p not in used_pairs for p in pairs):
                selected.append(t)
                for p in pairs:
                    used_pairs.add(p)
        if len(selected) >= 10:
            candidates.append(selected[:])
    
    # Also try a structured approach: start with a known good partial STS
    # and extend
    if not candidates:
        candidates.append([(i*3, i*3+1, i*3+2) for i in range(5)])
    
    # Phase 2: For each candidate, evaluate with fast_alpha, keep best few
    scored = []
    for sys in candidates:
        masks = [(1<<t[0])|(1<<t[1])|(1<<t[2]) for t in sys]
        a = fast_alpha(masks)
        scored.append((a, sys))
    
    scored.sort(key=lambda x: x[0])
    
    # Keep top 5 diverse candidates
    top_candidates = []
    for a, sys in scored:
        if len(top_candidates) >= 5:
            break
        # Check diversity: not too similar to existing top candidates
        is_diverse = True
        for _, existing in top_candidates:
            if len(set(sys) & set(existing)) > 0.8 * len(sys):
                is_diverse = False
                break
        if is_diverse:
            top_candidates.append((a, sys))
    
    if not top_candidates:
        top_candidates = scored[:5]
    
    # Phase 3: Local search with exact alpha verification
    # Use swap moves, but verify with exact_alpha periodically
    best_system = top_candidates[0][1][:]
    best_alpha = top_candidates[0][0]
    
    # Precompute for speed
    def get_used_pairs(system):
        pairs = set()
        for t in system:
            pairs.add((t[0],t[1]))
            pairs.add((t[0],t[2]))
            pairs.add((t[1],t[2]))
        return pairs
    
    current_system = best_system[:]
    current_masks = [(1<<t[0])|(1<<t[1])|(1<<t[2]) for t in current_system]
    current_alpha = best_alpha
    
    T = 3.0
    cooling = 0.9995
    min_T = 0.05
    
    while time.time() - start_time < DEADLINE:
        if T < min_T:
            break
        
        if not current_system:
            break
        
        # Generate move: swap one triple
        idx = random.randint(0, len(current_system)-1)
        old_t = current_system[idx]
        
        # Remove old triple
        temp_system = current_system[:]
        temp_system.pop(idx)
        
        used_pairs = get_used_pairs(temp_system)
        
        # Find compatible replacement
        new_t = None
        for _ in range(30):
            cand = random.choice(all_triples)
            pairs = [(cand[0],cand[1]), (cand[0],cand[2]), (cand[1],cand[2])]
            if all(p not in used_pairs for p in pairs):
                new_t = cand
                break
        
        if new_t is None:
            T *= cooling
            continue
        
        # Apply move
        current_system[idx] = new_t
        current_masks = [(1<<t[0])|(1<<t[1])|(1<<t[2]) for t in current_system]
        
        # Evaluate with fast alpha
        new_alpha = fast_alpha(current_masks)
        delta = new_alpha - current_alpha
        
        if delta < 0:
            current_alpha = new_alpha
            if new_alpha < best_alpha:
                best_alpha = new_alpha
                best_system = current_system[:]
        elif delta == 0:
            current_alpha = new_alpha
            # With some probability, accept and try to refine
            if random.random() < 0.5:
                current_alpha = new_alpha
        else:
            if delta <= 5 * T:
                if random.random() < pow(2.718281828, -delta / T):
                    current_alpha = new_alpha
            else:
                # Revert
                current_system[idx] = old_t
                current_masks = [(1<<t[0])|(1<<t[1])|(1<<t[2]) for t in current_system]
        
        T *= cooling
    
    # Final: try exact alpha on best_system to confirm, and do a final local polish
    # Do a few rounds of exact verification on best_system
    best_masks = [(1<<t[0])|(1<<t[1])|(1<<t[2]) for t in best_system]
    
    # Try to improve best_system with targeted swaps using exact alpha
    # But exact_alpha is expensive, so only do it a few times
    for _ in range(50):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        # Try a random swap on best_system
        test_system = best_system[:]
        idx = random.randint(0, len(test_system)-1)
        old_t = test_system[idx]
        temp = test_system[:]
        temp.pop(idx)
        used_pairs = get_used_pairs(temp)
        
        new_t = None
        for _ in range(20):
            cand = random.choice(all_triples)
            pairs = [(cand[0],cand[1]), (cand[0],cand[2]), (cand[1],cand[2])]
            if all(p not in used_pairs for p in pairs):
                new_t = cand
                break
        
        if new_t is None:
            continue
        
        test_system[idx] = new_t
        test_masks = [(1<<t[0])|(1<<t[1])|(1<<t[2]) for t in test_system]
        
        # Use exact alpha
        test_alpha = exact_alpha(test_masks)
        
        if test_alpha < best_alpha:
            best_alpha = test_alpha
            best_system = test_system[:]
            best_masks = test_masks
    
    # Final verification
    if not check_linearity(best_system):
        # Fallback to a simple valid system
        best_system = [(i*3, i*3+1, i*3+2) for i in range(5)]
    
    return best_system

if __name__ == "__main__":
    sys = construct()
    print(f"System size: {len(sys)}")
    for i in range(len(sys)):
        for j in range(i+1, len(sys)):
            common = set(sys[i]) & set(sys[j])
            if len(common) > 1:
                print("Invalid system!")
                break
