"""Search for ambient 12-set minimizing largest Sidon subset size."""
import random
import time
from itertools import combinations

def _is_sidon_fast(S):
    """Check if S is a Sidon set (all pair sums including diagonals are unique)."""
    n = len(S)
    seen = set()
    for i in range(n):
        si = S[i]
        for j in range(i, n):
            s = si + S[j]
            if s in seen:
                return False
            seen.add(s)
    return True

def _max_sidon_size(A, timeout=2.0):
    """Find the maximum Sidon subset size of A. Returns None if timeout."""
    n = len(A)
    t0 = time.time()
    
    # Check full set
    if _is_sidon_fast(A):
        return n
    
    # Try size 11: skip one element
    for skip in range(n):
        if time.time() - t0 > timeout:
            return None
        cand = A[:skip] + A[skip+1:]
        if _is_sidon_fast(cand):
            return 11
    
    # Try size 10: skip two elements
    for skip1 in range(n):
        for skip2 in range(skip1 + 1, n):
            if time.time() - t0 > timeout:
                return None
            cand = [x for i, x in enumerate(A) if i != skip1 and i != skip2]
            if _is_sidon_fast(cand):
                return 10
    
    # Try size 9: skip three elements
    for skip1 in range(n):
        for skip2 in range(skip1 + 1, n):
            for skip3 in range(skip2 + 1, n):
                if time.time() - t0 > timeout:
                    return None
                cand = [x for i, x in enumerate(A) if i != skip1 and i != skip2 and i != skip3]
                if _is_sidon_fast(cand):
                    return 9
    
    return 8

def _count_sum_collisions(A):
    """Count how many pair sums collide. Higher = harder to extract large Sidon set."""
    n = len(A)
    sums = {}
    for i in range(n):
        for j in range(i, n):
            s = A[i] + A[j]
            sums[s] = sums.get(s, 0) + 1
    # Count pairs that share a sum value
    collisions = 0
    for count in sums.values():
        if count > 1:
            collisions += count - 1
    return collisions

def construct():
    deadline = time.time() + 85.0  # Safety margin before 90s kill

    def _validate(A):
        if len(A) != 12:
            return False
        if len(set(A)) != 12:
            return False
        if any(x < 0 or x > 128 for x in A):
            return False
        return True

    candidates = []
    best_ms = 12  # Track best (lowest) max Sidon size found

    # Strategy 1: Clusters of 3 close elements
    # Within cluster {a, a+1, a+2}: collision a+(a+2) = (a+1)+(a+1)
    # 4 clusters of 3 => guaranteed max Sidon <= 8
    for gap in [8, 10, 12, 15, 18, 20, 25, 30, 35, 40, 50]:
        max_offset = 129 - 2 - 3 * gap
        for offset in range(0, max(1, max_offset), 1):
            A = []
            valid = True
            for k in range(4):
                base = offset + k * gap
                if base + 2 > 128:
                    valid = False
                    break
                A.extend([base, base + 1, base + 2])
            if valid and len(A) == 12 and len(set(A)) == 12:
                # Quick collision check before expensive Sidon test
                coll = _count_sum_collisions(A)
                if coll > 5:  # High collision count likely means low max Sidon
                    ms = _max_sidon_size(A, timeout=0.3)
                    if ms is not None:
                        candidates.append((A, ms))
                        if ms < best_ms:
                            best_ms = ms
                elif time.time() < deadline - 10:
                    ms = _max_sidon_size(A, timeout=0.3)
                    if ms is not None:
                        candidates.append((A, ms))
                        if ms < best_ms:
                            best_ms = ms

    # Strategy 2: Pairs of close elements (6 pairs)
    for gap in [4, 5, 6, 7, 8, 10, 12, 15, 20]:
        max_offset = 129 - 1 - 6 * gap
        for offset in range(0, max(1, max_offset), 1):
            A = []
            valid = True
            for k in range(6):
                base = offset + k * gap
                if base + 1 > 128:
                    valid = False
                    break
                A.extend([base, base + 1])
            if valid and len(A) == 12 and len(set(A)) == 12:
                coll = _count_sum_collisions(A)
                if coll > 4 or time.time() < deadline - 10:
                    ms = _max_sidon_size(A, timeout=0.3)
                    if ms is not None:
                        candidates.append((A, ms))
                        if ms < best_ms:
                            best_ms = ms

    # Strategy 3: Mixed cluster sizes - some triples, some pairs
    # e.g., 2 triples + 3 pairs, or 1 triple + 4 pairs + 2 singles
    for triple_count in [1, 2, 3]:
        for pair_count in [3, 4, 5]:
            if triple_count * 3 + pair_count * 2 != 12:
                continue
            for gap_t in [10, 15, 20, 30]:
                for gap_p in [5, 8, 10, 15, 20]:
                    for offset in range(0, 50, 2):
                        A = []
                        pos = offset
                        valid = True
                        for k in range(triple_count):
                            if pos + 2 > 128:
                                valid = False
                                break
                            A.extend([pos, pos + 1, pos + 2])
                            pos += gap_t
                        for k in range(pair_count):
                            if pos + 1 > 128:
                                valid = False
                                break
                            A.extend([pos, pos + 1])
                            pos += gap_p
                        if valid and len(A) == 12 and len(set(A)) == 12:
                            coll = _count_sum_collisions(A)
                            if coll > 3 or time.time() < deadline - 8:
                                ms = _max_sidon_size(A, timeout=0.25)
                                if ms is not None:
                                    candidates.append((A, ms))
                                    if ms < best_ms:
                                        best_ms = ms
                    if time.time() > deadline - 15:
                        break
                if time.time() > deadline - 15:
                    break
            if time.time() > deadline - 15:
                break
        if time.time() > deadline - 15:
            break

    # Strategy 4: Random search with collision-guided sampling
    rng = random.Random(42)
    for _ in range(50000):
        if time.time() > deadline - 5.0:
            break
        # Mix of random and structured
        if rng.random() < 0.5:
            # Random
            A = sorted(rng.sample(range(0, 129), 12))
        else:
            # Structured: pick cluster centers, then add offsets
            num_clusters = rng.randint(3, 6)
            centers = sorted(rng.sample(range(5, 124), num_clusters))
            A = []
            for c in centers:
                cluster_size = rng.randint(2, 3)
                for d in range(cluster_size):
                    val = c + d
                    if 0 <= val <= 128 and val not in A:
                        A.append(val)
            # Fill remaining with random
            while len(A) < 12:
                val = rng.randint(0, 128)
                if val not in A:
                    A.append(val)
            A = sorted(A[:12])
        
        if len(A) == 12 and len(set(A)) == 12:
            coll = _count_sum_collisions(A)
            if coll > 4 or time.time() < deadline - 10:
                ms = _max_sidon_size(A, timeout=0.2)
                if ms is not None:
                    candidates.append((A, ms))
                    if ms < best_ms:
                        best_ms = ms

    # Strategy 5: Local search - perturb best candidates
    if candidates and time.time() < deadline - 8:
        candidates.sort(key=lambda t: t[1])
        for A, ms in candidates[:20]:
            if time.time() > deadline - 5:
                break
            if ms >= 8:
                continue
            # Try small perturbations
            for _ in range(50):
                if time.time() > deadline - 5:
                    break
                A_new = A[:]
                idx = rng.randint(0, 11)
                A_new[idx] = rng.randint(0, 128)
                A_new = sorted(A_new)
                if len(set(A_new)) != 12:
                    continue
                coll = _count_sum_collisions(A_new)
                if coll > 4:
                    ms_new = _max_sidon_size(A_new, timeout=0.15)
                    if ms_new is not None and ms_new < best_ms:
                        best_ms = ms_new
                        candidates.append((A_new, ms_new))

    if not candidates:
        # Fallback: cluster of 3s
        A = [0, 1, 2, 10, 11, 12, 20, 21, 22, 30, 31, 32]
        if _validate(A):
            return A
        return [3 * i for i in range(12)]

    candidates.sort(key=lambda t: t[1])
    best = candidates[0][0]

    if not _validate(best):
        return [3 * i for i in range(12)]

    return best
