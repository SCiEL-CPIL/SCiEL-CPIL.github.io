"""Covering code K_3(6,1): minimum dominating set on H(6,3) via structured local search."""
import time
import random
import itertools
import numpy as np

_start = time.time()
_DEADLINE = 140.0  # safety margin under the 150s wall-clock kill


def _elapsed():
    return time.time() - _start


def _tetracode():
    # 9 words of length 4: a perfect code of radius 1 in H(4,3) (covers all 81 vertices).
    return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]


def _greedy_cover(q, n, radius):
    # Fast valid (non-optimal) covering code: greedy set cover over the whole space.
    from itertools import product
    space = np.array(list(product(range(q), repeat=n)), dtype=np.int8)
    N = space.shape[0]
    unc_set = set(range(N))
    chosen = []
    space_i = space
    while unc_set and _elapsed() < _DEADLINE - 2:
        best = -1
        best_cnt = 0
        for ci in range(N):
            c = space_i[ci]
            diff = (space_i != c).sum(axis=1)
            cov = np.nonzero(diff <= radius)[0]
            if len(cov) == 0:
                continue
            cnt = 0
            for v in cov:
                if v in unc_set:
                    cnt += 1
            if cnt > best_cnt:
                best_cnt = cnt
                best = ci
                if cnt == len(unc_set):
                    break
        if best < 0 or best_cnt == 0:
            rem = list(unc_set)
            if not rem:
                break
            best = rem[0]
            best_cnt = 1
        chosen.append(list(space_i[best]))
        c = space_i[best]
        diff = (space_i != c).sum(axis=1)
        for v in np.nonzero(diff <= radius)[0]:
            unc_set.discard(int(v))
    if unc_set:
        for v in list(unc_set):
            chosen.append(list(space_i[v]))
    return chosen


def _solve_q3_n6_r1(deadline):
    """Structured local search for K_3(6,1): start from 81-word product code, reduce to 73."""
    q = 3
    n = 6
    radius = 1
    total_words = 729
    
    # Precompute all words and index map
    all_tuples = list(itertools.product(range(q), repeat=n))
    idx_map = {t: i for i, t in enumerate(all_tuples)}
    
    # Precompute neighbor bitsets (radius 1 ball)
    neighbors = [0] * total_words
    neighbor_indices = [[] for _ in range(total_words)]
    for i, t in enumerate(all_tuples):
        mask = 1 << i
        neighbor_indices[i].append(i)
        for pos in range(n):
            orig = t[pos]
            for new_val in range(q):
                if new_val == orig:
                    continue
                t_list = list(t)
                t_list[pos] = new_val
                nt = tuple(t_list)
                j = idx_map[nt]
                mask |= (1 << j)
                neighbor_indices[i].append(j)
        neighbors[i] = mask
    
    full_mask = (1 << total_words) - 1
    
    # Precompute perturbation vectors in Z_3^6
    perturbations = []
    for pos in range(n):
        for delta in (1, 2):
            p = [0] * n
            p[pos] = delta
            perturbations.append(p)
    for pos1 in range(n):
        for pos2 in range(pos1 + 1, n):
            for d1 in (1, 2):
                for d2 in (1, 2):
                    p = [0] * n
                    p[pos1] = d1
                    p[pos2] = d2
                    perturbations.append(p)
    
    # Start from the 81-word product code (tetracode x identity on 2 coords)
    tetra = _tetracode()
    base_indices = set()
    for h in tetra:
        for tail in itertools.product(range(3), repeat=2):
            w = h + list(tail)
            base_indices.add(idx_map[tuple(w)])
    
    code_set = set(base_indices)
    cover_count = [0] * total_words
    for c in code_set:
        for idx in neighbor_indices[c]:
            cover_count[idx] += 1
    
    # Verify initial coverage
    uncovered_mask = 0
    for i in range(total_words):
        if cover_count[i] == 0:
            uncovered_mask |= (1 << i)
    if uncovered_mask != 0:
        # Should not happen, but fix it
        while uncovered_mask:
            best_idx = -1
            best_cnt = -1
            for i in range(total_words):
                if i in code_set:
                    continue
                new_covered = neighbors[i] & uncovered_mask
                cnt = new_covered.bit_count()
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_idx = i
            if best_idx == -1:
                break
            code_set.add(best_idx)
            for idx in neighbor_indices[best_idx]:
                cover_count[idx] += 1
            uncovered_mask &= ~neighbors[best_idx]
    
    # Phase 1: Aggressive removal of redundant codewords
    def try_removals(code_set, cover_count, max_rounds=50):
        improved = True
        rounds = 0
        while improved and rounds < max_rounds and _elapsed() < deadline - 5:
            improved = False
            rounds += 1
            # Shuffle order for diversity
            order = list(code_set)
            random.shuffle(order)
            for c in order:
                if c not in code_set:
                    continue
                # Check if c is redundant: all its neighbors have cover_count >= 2
                redundant = True
                for idx in neighbor_indices[c]:
                    if cover_count[idx] == 1:
                        redundant = False
                        break
                if redundant:
                    code_set.remove(c)
                    for idx in neighbor_indices[c]:
                        cover_count[idx] -= 1
                    improved = True
                    break  # Restart after modification
        return code_set, cover_count
    
    code_set, cover_count = try_removals(code_set, cover_count)
    
    # Phase 2: Multi-move local search
    rng = random.Random(42)
    
    def _try_remove_one_add_two(code_set, cover_count, rng, max_attempts=300):
        """Try removing one codeword and adding two nearby codewords to enable reduction."""
        improved = False
        for _ in range(max_attempts):
            if _elapsed() > deadline - 5:
                break
            if len(code_set) < 2:
                break
            c = rng.choice(list(code_set))
            c_tuple = all_tuples[c]
            
            # Pick two perturbation vectors
            d1 = rng.choice(perturbations)
            d2 = rng.choice(perturbations)
            
            t1 = tuple((c_tuple[i] + d1[i]) % q for i in range(n))
            t2 = tuple((c_tuple[i] + d2[i]) % q for i in range(n))
            n1 = idx_map[t1]
            n2 = idx_map[t2]
            
            if n1 in code_set or n2 in code_set or n1 == n2:
                continue
            
            # Remove c
            for idx in neighbor_indices[c]:
                cover_count[idx] -= 1
            
            # Find uncovered after removal
            uncovered = 0
            for idx in neighbor_indices[c]:
                if cover_count[idx] == 0:
                    uncovered |= (1 << idx)
            
            if uncovered == 0:
                code_set.remove(c)
                improved = True
                break
            
            # Add n1 and n2
            code_set.remove(c)
            code_set.add(n1)
            code_set.add(n2)
            for idx in neighbor_indices[n1]:
                cover_count[idx] += 1
            for idx in neighbor_indices[n2]:
                cover_count[idx] += 1
            
            # Check if uncovered is fixed
            still_uncovered = 0
            for idx in neighbor_indices[c]:
                if cover_count[idx] == 0:
                    still_uncovered |= (1 << idx)
            
            if still_uncovered == 0:
                # Check global coverage
                all_ok = True
                for i in range(total_words):
                    if cover_count[i] == 0:
                        all_ok = False
                        break
                if all_ok:
                    improved = True
                    break
                else:
                    code_set.remove(n1)
                    code_set.remove(n2)
                    for idx in neighbor_indices[n1]:
                        cover_count[idx] -= 1
                    for idx in neighbor_indices[n2]:
                        cover_count[idx] -= 1
                    code_set.add(c)
                    for idx in neighbor_indices[c]:
                        cover_count[idx] += 1
            else:
                code_set.remove(n1)
                code_set.remove(n2)
                for idx in neighbor_indices[n1]:
                    cover_count[idx] -= 1
                for idx in neighbor_indices[n2]:
                    cover_count[idx] -= 1
                code_set.add(c)
                for idx in neighbor_indices[c]:
                    cover_count[idx] += 1
        
        return code_set, cover_count, improved
    
    def _try_two_for_two_swap(code_set, cover_count, rng, max_attempts=200):
        """Try swapping two codewords for two others to enable further reductions."""
        improved = False
        for _ in range(max_attempts):
            if _elapsed() > deadline - 5:
                break
            if len(code_set) < 2:
                break
            c1, c2 = rng.sample(list(code_set), 2)
            d1 = rng.choice(perturbations)
            d2 = rng.choice(perturbations)
            
            t1 = all_tuples[c1]
            t2 = all_tuples[c2]
            n1 = idx_map[tuple((t1[i] + d1[i]) % q for i in range(n))]
            n2 = idx_map[tuple((t2[i] + d2[i]) % q for i in range(n))]
            
            if n1 in code_set or n2 in code_set or n1 == c1 or n1 == c2 or n2 == c1 or n2 == c2:
                continue
            
            # Remove c1 and c2
            for idx in neighbor_indices[c1]:
                cover_count[idx] -= 1
            for idx in neighbor_indices[c2]:
                cover_count[idx] -= 1
            
            # Check uncovered
            uncovered = 0
            for idx in neighbor_indices[c1] + neighbor_indices[c2]:
                if cover_count[idx] == 0:
                    uncovered |= (1 << idx)
            
            if uncovered == 0:
                code_set.remove(c1)
                code_set.remove(c2)
                improved = True
                break
            
            # Add n1 and n2
            code_set.remove(c1)
            code_set.remove(c2)
            code_set.add(n1)
            code_set.add(n2)
            for idx in neighbor_indices[n1]:
                cover_count[idx] += 1
            for idx in neighbor_indices[n2]:
                cover_count[idx] += 1
            
            # Check if all covered
            all_ok = True
            for i in range(total_words):
                if cover_count[i] == 0:
                    all_ok = False
                    break
            
            if all_ok:
                improved = True
                break
            else:
                code_set.remove(n1)
                code_set.remove(n2)
                for idx in neighbor_indices[n1]:
                    cover_count[idx] -= 1
                for idx in neighbor_indices[n2]:
                    cover_count[idx] -= 1
                code_set.add(c1)
                code_set.add(c2)
                for idx in neighbor_indices[c1]:
                    cover_count[idx] += 1
                for idx in neighbor_indices[c2]:
                    cover_count[idx] += 1
        
        return code_set, cover_count, improved
    
    # Interleave removals and multi-move swaps
    for round_num in range(40):
        if _elapsed() > deadline - 5:
            break
        
        # Try various multi-move operations
        code_set, cover_count, _ = _try_remove_one_add_two(code_set, cover_count, rng, max_attempts=100)
        code_set, cover_count, _ = _try_two_for_two_swap(code_set, cover_count, rng, max_attempts=80)
        
        # Try removals
        code_set, cover_count = try_removals(code_set, cover_count, max_rounds=30)
        
        # If we've reached 73 or fewer, we're done
        if len(code_set) <= 73:
            break
    
    # Phase 3: Final aggressive removal pass with additional multi-move attempts
    code_set, cover_count = try_removals(code_set, cover_count, max_rounds=100)
    
    # Extra multi-move rounds if time permits
    if len(code_set) > 73 and _elapsed() < deadline - 10:
        for _ in range(10):
            if _elapsed() > deadline - 5:
                break
            code_set, cover_count, _ = _try_remove_one_add_two(code_set, cover_count, rng, max_attempts=50)
            code_set, cover_count = try_removals(code_set, cover_count, max_rounds=20)
            if len(code_set) <= 73:
                break
    
    # Verify final coverage
    uncovered_mask = 0
    for i in range(total_words):
        if cover_count[i] == 0:
            uncovered_mask |= (1 << i)
    
    if uncovered_mask != 0:
        # Repair any uncovered vertices
        while uncovered_mask and _elapsed() < deadline - 2:
            best_idx = -1
            best_cnt = -1
            for i in range(total_words):
                if i in code_set:
                    continue
                new_covered = neighbors[i] & uncovered_mask
                cnt = new_covered.bit_count()
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_idx = i
            if best_idx == -1:
                break
            code_set.add(best_idx)
            for idx in neighbor_indices[best_idx]:
                cover_count[idx] += 1
            uncovered_mask &= ~neighbors[best_idx]
    
    # Convert to list of lists
    result = [list(all_tuples[c]) for c in code_set]
    return result


def construct(q, n, radius):
    # Fast path: structured local search for the specific case q=3, n=6, radius=1.
    if q == 3 and n == 6 and radius == 1:
        result = None
        if _elapsed() < _DEADLINE - 10:
            try:
                result = _solve_q3_n6_r1(_DEADLINE)
            except Exception:
                result = None
        
        # Verify validity
        if result is not None:
            # Check all distinct
            if len(set(tuple(w) for w in result)) != len(result):
                result = None
            # Check coverage
            if result is not None:
                from itertools import product
                space = np.array(list(product(range(3), repeat=6)), dtype=np.int8)
                code_arr = np.array(result, dtype=np.int8)
                ok = True
                for v in space:
                    diff = (code_arr != v).sum(axis=1)
                    if diff.min() > 1:
                        ok = False
                        break
                if not ok:
                    result = None
        
        if result is not None:
            return result
        # Fallback: greedy (valid but suboptimal)
        return _greedy_cover(q, n, radius)
    else:
        # Generic: greedy
        return _greedy_cover(q, n, radius)
