"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    def count_nonzero(poly):
        return int(np.count_nonzero(poly))

    def poly_square_from_dense(dense):
        if len(dense) == 0:
            return np.array([0], dtype=np.int64)
        return np.convolve(dense, dense, mode='full')

    def terms_to_dense(terms):
        if not terms:
            return np.array([0], dtype=np.int64)
        max_deg = max(pos for pos, _ in terms)
        dense = np.zeros(max_deg + 1, dtype=np.int64)
        for pos, val in terms:
            dense[pos] = val
        return dense

    def evaluate(terms):
        if len(terms) != 12:
            return 1000
        positions = [pos for pos, _ in terms]
        if len(set(positions)) != 12:
            return 1000
        for pos, val in terms:
            if val == 0 or abs(val) > 64 or pos > 40:
                return 1000
        dense = terms_to_dense(terms)
        sq = poly_square_from_dense(dense)
        return count_nonzero(sq)

    def sum_collision_count(positions):
        """Count distinct sum positions p_i + p_j for i <= j"""
        sums = set()
        for i in range(12):
            for j in range(i, 12):
                sums.add(positions[i] + positions[j])
        return len(sums)

    def find_best_values_for_positions(positions, time_limit=1.0):
        """For fixed positions, find values minimizing square nonzero count."""
        start_t = time.time()
        best_score = 1000
        best_values = None
        
        # Build sum-to-pairs mapping
        sum_to_pairs = defaultdict(list)
        for i in range(12):
            for j in range(i, 12):
                s = positions[i] + positions[j]
                sum_to_pairs[s].append((i, j))
        
        # Try random value assignments with bias toward small values and signs
        n_trials = 0
        while time.time() - start_t < time_limit and n_trials < 50000:
            n_trials += 1
            # Use small values to increase chance of cancellation
            vals = []
            for _ in range(12):
                sign = random.choice([1, -1])
                # Bias toward small values
                r = random.random()
                if r < 0.3:
                    mag = 1
                elif r < 0.6:
                    mag = random.randint(1, 3)
                elif r < 0.8:
                    mag = random.randint(1, 8)
                else:
                    mag = random.randint(1, 64)
                vals.append(sign * mag)
            
            dense = np.zeros(max(positions) + 1, dtype=np.int64)
            for p, v in zip(positions, vals):
                dense[p] = v
            sq = np.convolve(dense, dense, mode='full')
            score = count_nonzero(sq)
            if score < best_score:
                best_score = score
                best_values = vals[:]
                if best_score <= 12:
                    break
        
        return best_score, best_values

    def structural_search():
        """Enumerate position sets with high sum-collision density."""
        best_terms = None
        best_score = 1000
        
        # Generate candidate position sets with high collision density
        # Key: positions that are arithmetic progressions or have regular structure
        candidates = []
        
        # Arithmetic progressions
        for start in range(0, 41):
            for step in range(1, 5):
                pos = [start + i * step for i in range(12)]
                if pos[-1] <= 40:
                    candidates.append(tuple(pos))
        
        # Two interleaved arithmetic progressions
        for s1 in range(0, 30):
            for s2 in range(s1 + 1, 30):
                for step in range(1, 5):
                    pos = []
                    for i in range(6):
                        pos.append(s1 + i * step)
                        pos.append(s2 + i * step)
                    pos.sort()
                    if len(set(pos)) == 12 and pos[-1] <= 40:
                        candidates.append(tuple(pos))
        
        # Clusters
        for n_clusters in [2, 3, 4, 6]:
            if 12 % n_clusters != 0:
                continue
            cluster_size = 12 // n_clusters
            for _ in range(200):
                bases = sorted(random.sample(range(0, 41 - cluster_size + 1), n_clusters))
                pos = []
                for b in bases:
                    offsets = random.sample(range(0, cluster_size + 2), cluster_size)
                    pos.extend(b + o for o in offsets)
                pos = sorted(set(pos))
                if len(pos) == 12 and pos[-1] <= 40:
                    candidates.append(tuple(pos))
        
        # Random sets
        for _ in range(500):
            pos = tuple(sorted(random.sample(range(41), 12)))
            candidates.append(pos)
        
        # Sort by sum collision count (fewer distinct sums = more potential cancellation)
        candidates = list(set(candidates))
        candidates.sort(key=lambda p: sum_collision_count(p))
        
        # Try top candidates
        for idx, positions in enumerate(candidates[:200]):
            if time.time() > deadline - 5:
                break
            
            # Estimate time per candidate
            remaining = deadline - time.time()
            n_remaining = len(candidates) - idx
            if n_remaining > 0:
                time_per = max(0.05, remaining / n_remaining)
            else:
                time_per = 0.5
            
            score, values = find_best_values_for_positions(list(positions), time_limit=time_per)
            
            if values is not None and score < best_score:
                best_score = score
                best_terms = list(zip(positions, values))
                if best_score <= 12:
                    break
        
        return best_terms, best_score

    def local_search_from(terms, time_limit=2.0):
        """Local search starting from given terms."""
        start_t = time.time()
        current_terms = list(terms)
        current_score = evaluate(current_terms)
        
        def neighbor(terms):
            terms = list(terms)
            move_type = random.random()
            if move_type < 0.4:
                idx = random.randint(0, 11)
                old_pos = terms[idx][0]
                new_pos = random.randint(0, 40)
                if new_pos == old_pos:
                    return terms
                for i, (pos, val) in enumerate(terms):
                    if i != idx and pos == new_pos:
                        return terms
                terms[idx] = (new_pos, terms[idx][1])
            elif move_type < 0.7:
                idx = random.randint(0, 11)
                new_val = random.choice([1, -1]) * random.randint(1, 64)
                terms[idx] = (terms[idx][0], new_val)
            elif move_type < 0.9:
                idx = random.randint(0, 11)
                terms[idx] = (terms[idx][0], -terms[idx][1])
            else:
                i, j = random.sample(range(12), 2)
                terms[i], terms[j] = (terms[j][0], terms[i][1]), (terms[i][0], terms[j][1])
            return terms
        
        while time.time() - start_t < time_limit:
            improved = False
            for _ in range(50):
                if time.time() - start_t > time_limit:
                    break
                new_terms = neighbor(current_terms)
                new_score = evaluate(new_terms)
                if new_score < current_score:
                    current_terms = new_terms
                    current_score = new_score
                    improved = True
            if not improved:
                break
        
        return current_terms, current_score

    # Phase 1: Structural search
    st_terms, st_score = structural_search()
    
    best_terms = st_terms
    best_score = st_score
    
    # Phase 2: Local search refinement
    if best_terms is not None:
        remaining = deadline - time.time()
        if remaining > 3:
            refined, refined_score = local_search_from(best_terms, time_limit=min(remaining - 2, 10.0))
            if refined_score < best_score:
                best_terms = refined
                best_score = refined_score
    
    # Phase 3: Random restarts with local search
    remaining = deadline - time.time()
    if remaining > 5:
        n_restarts = min(50, int(remaining / 2))
        for _ in range(n_restarts):
            if time.time() > deadline - 3:
                break
            positions = random.sample(range(41), 12)
            positions.sort()
            values = [random.choice([1, -1]) * random.randint(1, 64) for _ in range(12)]
            terms = list(zip(positions, values))
            
            # Quick local search
            t_limit = max(0.5, (deadline - 3 - time.time()) / max(1, n_restarts - _))
            refined, score = local_search_from(terms, time_limit=t_limit)
            if score < best_score:
                best_score = score
                best_terms = refined
                if best_score <= 12:
                    break
    
    if best_terms is None:
        # Fallback
        positions = list(range(0, 41, 4))[:12]
        values = [1] * 12
        best_terms = list(zip(positions, values))
        best_score = evaluate(best_terms)
    
    max_pos = max(pos for pos, _ in best_terms)
    coeff_list = [0] * (max_pos + 1)
    for pos, val in best_terms:
        coeff_list[pos] = val
    
    if len(coeff_list) > 41:
        coeff_list = coeff_list[:41]
    if coeff_list[-1] == 0:
        while coeff_list and coeff_list[-1] == 0:
            coeff_list.pop()
        if not coeff_list:
            coeff_list = [1]
    
    return coeff_list
# EVOLVE-BLOCK-END
