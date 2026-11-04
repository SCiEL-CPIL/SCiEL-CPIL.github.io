import time
import numpy as np

def _score_arr(U):
    if len(U) < 2:
        return 0.0
    mx = int(U[-1])
    if mx <= 0:
        return 0.0
    s = np.unique(U[:, None] + U[None, :])
    d = np.unique(U[:, None] - U[None, :])
    ratio = len(d) / len(s)
    if ratio <= 1.0:
        return 1.0
    return float(1.0 + np.log2(ratio) / np.log2(2*mx + 1))

def _score_set(Uset):
    U = np.array(sorted(Uset), dtype=np.int64)
    return _score_arr(U)

def _generate_candidates(U, limit_val):
    """
    Generate candidate elements for addition based on sums and differences of existing elements.
    This is the 'structural idea' from the inspiration.
    """
    U_set = set(U)
    candidates = set()
    
    for a in U:
        for b in U:
            s = a + b
            d = abs(a - b)
            if 0 < s <= limit_val and s not in U_set:
                candidates.add(s)
            if 0 < d <= limit_val and d not in U_set:
                candidates.add(d)
            
    return list(candidates)

def construct():
    deadline = time.time() + 110
    rng = np.random.default_rng(12345)
    
    # Initial best set
    best = [0, 3, 9, 27]
    best_score = _score_set(set(best))
    
    M_max = 500  # Increased max value for more exploration
    
    def try_update(trial):
        nonlocal best, best_score
        if 0 not in trial or len(trial) < 2:
            return
        sc = _score_set(trial)
        if sc > best_score:
            best_score = sc
            best = list(trial)
    
    def greedy(cur, cur_score, budget):
        """
        Local search with dynamic candidate generation.
        Combines the 'refinement machinery' of the current program (greedy steps)
        with the 'structural idea' of the inspiration (candidate generation).
        """
        nonlocal best, best_score
        improved = True
        no_improvement_count = 0
        max_no_improvement = 20
        
        # Initial candidate pool
        current_max = max(cur) if cur else 0
        candidates = _generate_candidates(cur, max(M_max, current_max * 2))
        
        while time.time() < budget and no_improvement_count < max_no_improvement:
            improved = False
            
            # Update candidate pool periodically or if empty
            if not candidates or rng.random() < 0.1:
                current_max = max(cur) if cur else 0
                new_limit = max(M_max, current_max * 2)
                candidates = _generate_candidates(cur, new_limit)
            
            # 1. Try to ADD an element from candidates
            if len(cur) < 40 and candidates:
                # Try a few random candidates
                for _ in range(10):
                    if not candidates:
                        break
                    idx = rng.integers(0, len(candidates))
                    c = candidates.pop(idx)
                    if c in cur:
                        continue
                    trial = cur | {c}
                    sc = _score_set(trial)
                    if sc > cur_score + 1e-12:
                        cur = trial
                        cur_score = sc
                        improved = True
                        # Regenerate candidates immediately as structure changed significantly
                        current_max = max(cur)
                        candidates = _generate_candidates(cur, max(M_max, current_max * 2))
                        if sc > best_score:
                            best_score = sc
                            best = list(cur)
                        break
            
            # 2. Try to REMOVE an element
            if len(cur) > 2:
                removable = [x for x in cur if x != 0]
                if removable:
                    c = removable[rng.integers(0, len(removable))]
                    trial = cur - {c}
                    if 0 in trial and len(trial) >= 2:
                        sc = _score_set(trial)
                        if sc > cur_score + 1e-12:
                            cur = trial
                            cur_score = sc
                            improved = True
                            candidates = _generate_candidates(cur, M_max)
                            if sc > best_score:
                                best_score = sc
                                best = list(cur)
            
            # 3. Try to REPLACE an element
            if len(cur) > 1 and candidates:
                idx = rng.integers(1, len(cur))
                old_val = list(cur)[idx]
                # Pick a random candidate
                for _ in range(5):
                    if not candidates:
                        break
                    cidx = rng.integers(0, len(candidates))
                    new_val = candidates.pop(cidx)
                    if new_val == old_val or new_val in cur:
                        continue
                    trial = (cur - {old_val}) | {new_val}
                    if 0 not in trial:
                        continue
                    sc = _score_set(trial)
                    if sc > cur_score + 1e-12:
                        cur = trial
                        cur_score = sc
                        improved = True
                        current_max = max(cur)
                        candidates = _generate_candidates(cur, max(M_max, current_max * 2))
                        if sc > best_score:
                            best_score = sc
                            best = list(cur)
                        break
            
            if improved:
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                
        return cur, cur_score

    # Structured seeds from both current and inspiration
    seeds = [
        {0, 3, 9, 27},
        {0, 1, 3, 9, 27},
        {0, 2, 4, 8, 16, 32},
        {0, 3, 6, 9, 12, 15, 18},
        {0, 5, 10, 15, 20, 25},
        {0, 3, 9, 27, 81},
        {0, 1, 4, 9, 16, 25, 36, 49, 64, 81}, # Squares
        {0, 1, 2, 4, 8, 16, 32, 64, 128}, # Powers of 2
        {0, 1, 3, 7, 15, 31, 63, 127}, # Mersenne-like
        {0, 1, 4, 10, 20, 35, 56, 84}, # Triangular
    ]
    
    # Phase 1: Evaluate seeds and run greedy from the best ones
    seed_scores = []
    for seed in seeds:
        if time.time() > deadline - 60:
            break
        sc = _score_set(seed)
        seed_scores.append((sc, seed))
        if sc > best_score:
            best_score = sc
            best = list(seed)
            
    # Sort seeds by score and run greedy on top few
    seed_scores.sort(key=lambda x: x[0], reverse=True)
    
    for sc, seed in seed_scores[:5]:
        if time.time() > deadline - 35:
            break
        cur, cur_score = greedy(set(seed), sc, deadline - 30)
        if cur_score > best_score:
            best_score = cur_score
            best = list(cur)
            
    # Phase 2: Simulated Annealing to escape local optima
    if time.time() < deadline - 5:
        cur = set(best)
        cur_score = _score_set(cur)
        T = 0.08
        T_min = 1e-6
        cooling = 0.997
        while T > T_min and time.time() < deadline - 2:
            x = int(rng.integers(0, M_max + 1))
            if x in cur:
                trial = cur - {x}
                if not (0 in trial and len(trial) >= 2):
                    T *= cooling
                    continue
            else:
                trial = cur | {x}
            sc = _score_set(trial)
            delta = sc - cur_score
            if delta > 0 or (rng.random() < np.exp(delta / T)):
                cur = trial
                cur_score = sc
                if sc > best_score:
                    best_score = sc
                    best = list(cur)
            T *= cooling

    # Phase 3: Final greedy polish
    if time.time() < deadline - 1:
        cur = set(best)
        cur_score = _score_set(cur)
        greedy(cur, cur_score, deadline - 0.5)

    return best
