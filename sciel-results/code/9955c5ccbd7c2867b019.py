"""GRASP + Tabu Search for 3AP-maximizing 4AP-free subset of [1,256]."""
import time
import random
import numpy as np

N = 256
DEADLINE = 75.0  # safety margin under 90s wall-clock

def _has_4ap(A, x, d):
    """Check if {x, x+d, x+2d, x+3d} is a subset of A (all present)."""
    return (x in A and x + d in A and x + 2 * d in A and x + 3 * d in A)

def _count_3aps(A):
    """Count distinct 3-term APs (positive d, counted once)."""
    count = 0
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            if a in A and (a + d) in A and (a + 2 * d) in A:
                count += 1
    return count

def _count_3aps_fast(A, d_count):
    """Fast count using precomputed per-d difference counts."""
    total = 0
    for d in range(1, N // 2 + 1):
        total += d_count[d]
    return total

def _build_d_count(A):
    """Build array where d_count[d] = number of 3APs with difference d."""
    d_count = np.zeros(N, dtype=np.int64)
    A_list = sorted(A)
    A_set = set(A_list)
    for d in range(1, N // 2 + 1):
        cnt = 0
        for a in range(1, N - 2 * d + 1):
            if a in A_set and (a + d) in A_set and (a + 2 * d) in A_set:
                cnt += 1
        d_count[d] = cnt
    return d_count

def _violations(A):
    """Return list of (x, d) such that {x, x+d, x+2d, x+3d} ⊆ A."""
    v = []
    for d in range(1, (N - 1) // 3 + 1):
        for x in range(1, N - 3 * d + 1):
            if (x in A and x + d in A and x + 2 * d in A and x + 3 * d in A):
                v.append((x, d))
    return v

def _greedy_construct(rng):
    """Greedy: add elements that maximize 3AP gain while avoiding 4AP."""
    A = set()
    for x in range(1, N + 1):
        # Check if adding x creates a 4AP
        creates_4ap = False
        for d in range(1, (x - 1) // 3 + 1):
            if all((x - j * d) in A for j in (1, 2, 3)):
                creates_4ap = True
                break
        if not creates_4ap:
            # Check 4AP where x is not the start
            for d in range(1, (N - x) // 3 + 1):
                if all((x + j * d) in A for j in (1, 2, 3)):
                    creates_4ap = True
                    break
            if not creates_4ap:
                for d in range(1, (N - x) // 2 + 1):
                    if (x - d) in A and (x + d) in A and (x + 2 * d) in A:
                        creates_4ap = True
                        break
                    if (x - 2 * d) in A and (x - d) in A and (x + d) in A:
                        creates_4ap = True
                        break
        if not creates_4ap:
            A.add(x)
    return A

def _greedy_with_scores(rng):
    """Greedy with randomized scoring: prefer elements with high 3AP contribution."""
    A = set()
    candidates = list(range(1, N + 1))
    rng.shuffle(candidates)
    
    for x in candidates:
        # Check if adding x creates a 4AP
        creates_4ap = False
        for d in range(1, (x - 1) // 3 + 1):
            if all((x - j * d) in A for j in (1, 2, 3)):
                creates_4ap = True
                break
        if not creates_4ap:
            for d in range(1, (N - x) // 3 + 1):
                if all((x + j * d) in A for j in (1, 2, 3)):
                    creates_4ap = True
                    break
            if not creates_4ap:
                for d in range(1, (N - x) // 2 + 1):
                    if (x - d) in A and (x + d) in A and (x + 2 * d) in A:
                        creates_4ap = True
                        break
                    if (x - 2 * d) in A and (x - d) in A and (x + d) in A:
                        creates_4ap = True
                        break
        if not creates_4ap:
            A.add(x)
    return A

def _tabu_search(A, rng, time_limit):
    """Tabu search to improve 3AP count while maintaining 4AP-free."""
    A = set(A)
    d_count = _build_d_count(A)
    score = int(np.sum(d_count))
    best_score = score
    best_A = set(A)
    
    # Tabu list: (element, direction) -> expiration
    tabu = {}
    tabu_tenure = 15
    it = 0
    start_time = time.time()
    
    # Precompute for each element, which 4APs it's part of
    def _element_4ap_count(x, A_set):
        """Count 4APs containing x."""
        cnt = 0
        for d in range(1, (N - 1) // 3 + 1):
            for pos in range(4):
                base = x - pos * d
                if base < 1:
                    continue
                if base + 3 * d > N:
                    continue
                if all((base + j * d) in A_set for j in range(4)):
                    cnt += 1
        return cnt
    
    while time.time() - start_time < time_limit:
        it += 1
        # Generate candidate moves: add or remove an element
        moves = []
        
        # Removal moves
        for x in sorted(A):
            # Score delta: how many 3APs are lost
            lost = 0
            for d in range(1, (N - 1) // 2 + 1):
                # x as first term
                if (x + d) in A and (x + 2 * d) in A:
                    lost += 1
                # x as middle term
                if (x - d) in A and (x + d) in A:
                    lost += 1
                # x as last term
                if (x - d) in A and (x - 2 * d) in A:
                    lost += 1
            moves.append((-lost, 'remove', x))
        
        # Addition moves
        for x in range(1, N + 1):
            if x in A:
                continue
            # Check if adding x creates 4AP
            creates_4ap = False
            for d in range(1, (x - 1) // 3 + 1):
                if all((x - j * d) in A for j in (1, 2, 3)):
                    creates_4ap = True
                    break
            if not creates_4ap:
                for d in range(1, (N - x) // 3 + 1):
                    if all((x + j * d) in A for j in (1, 2, 3)):
                        creates_4ap = True
                        break
                if not creates_4ap:
                    for d in range(1, (N - x) // 2 + 1):
                        if (x - d) in A and (x + d) in A and (x + 2 * d) in A:
                            creates_4ap = True
                            break
                        if (x - 2 * d) in A and (x - d) in A and (x + d) in A:
                            creates_4ap = True
                            break
            if creates_4ap:
                continue
            # Score delta: how many 3APs are gained
            gained = 0
            for d in range(1, (N - 1) // 2 + 1):
                if (x + d) in A and (x + 2 * d) in A:
                    gained += 1
                if (x - d) in A and (x + d) in A:
                    gained += 1
                if (x - d) in A and (x - 2 * d) in A:
                    gained += 1
            moves.append((gained, 'add', x))
        
        if not moves:
            break
        
        # Sort by score delta (descending), with randomization
        moves.sort(key=lambda m: (-m[0], rng.random()))
        
        # Pick from top moves, avoiding tabu if possible
        best_move = None
        for delta, direction, x in moves[:20]:  # top 20
            key = (x, direction)
            if key in tabu and tabu[key] > it:
                continue
            best_move = (delta, direction, x)
            break
        if best_move is None:
            # All top moves are tabu; pick best non-tabu or best overall
            for delta, direction, x in moves:
                key = (x, direction)
                if key not in tabu or tabu[key] <= it:
                    best_move = (delta, direction, x)
                    break
            if best_move is None:
                best_move = moves[0]
        
        delta, direction, x = best_move
        
        # Apply move
        if direction == 'add':
            A.add(x)
        else:
            A.discard(x)
        
        # Update score
        score += delta
        tabu[(x, direction)] = it + tabu_tenure
        
        # Clean up expired tabu entries
        if it % 100 == 0:
            tabu = {k: v for k, v in tabu.items() if v > it}
        
        if score > best_score:
            best_score = score
            best_A = set(A)
    
    return best_A, best_score

def _local_search(A, rng, time_limit):
    """Simple local search: hill-climbing with random restarts."""
    A = set(A)
    score = _count_3aps(A)
    best_score = score
    best_A = set(A)
    start_time = time.time()
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        # Try all single-element moves
        best_delta = 0
        best_move = None
        
        # Try removals
        for x in list(A):
            A.discard(x)
            new_score = _count_3aps(A)
            delta = new_score - score
            if delta > best_delta:
                best_delta = delta
                best_move = ('remove', x)
            A.add(x)
        
        # Try additions
        for x in range(1, N + 1):
            if x in A:
                continue
            A.add(x)
            # Check 4AP
            ok = True
            for d in range(1, (x - 1) // 3 + 1):
                if all((x - j * d) in A for j in (1, 2, 3)):
                    ok = False
                    break
            if ok:
                for d in range(1, (N - x) // 3 + 1):
                    if all((x + j * d) in A for j in (1, 2, 3)):
                        ok = False
                        break
            if ok:
                for d in range(1, (N - x) // 2 + 1):
                    if (x - d) in A and (x + d) in A and (x + 2 * d) in A:
                        ok = False
                        break
                    if (x - 2 * d) in A and (x - d) in A and (x + d) in A:
                        ok = False
                        break
            if ok:
                new_score = _count_3aps(A)
                delta = new_score - score
                if delta > best_delta:
                    best_delta = delta
                    best_move = ('add', x)
            A.discard(x)
        
        if best_move and best_delta > 0:
            direction, x = best_move
            if direction == 'add':
                A.add(x)
            else:
                A.discard(x)
            score += best_delta
            if score > best_score:
                best_score = score
                best_A = set(A)
            improved = True
    
    return best_A, best_score

def construct():
    start_time = time.time()
    rng = random.Random(42)
    
    best_A = set()
    best_score = -1
    
    # Phase 1: Generate diverse starting solutions via GRASP
    grasp_time = min(15.0, DEADLINE - start_time)
    while time.time() - start_time < grasp_time:
        # Randomized greedy
        A = _greedy_with_scores(rng)
        score = _count_3aps(A)
        if score > best_score:
            best_score = score
            best_A = set(A)
        
        # Also try pure greedy
        A2 = _greedy_construct(rng)
        score2 = _count_3aps(A2)
        if score2 > best_score:
            best_score = score2
            best_A = set(A2)
    
    # Phase 2: Tabu search from best solution
    remaining = DEADLINE - start_time - 5.0
    if remaining > 10:
        best_A, best_score = _tabu_search(best_A, rng, remaining)
    
    # Phase 3: Final local search polish
    remaining = DEADLINE - start_time - 3.0
    if remaining > 2:
        best_A, best_score = _local_search(best_A, rng, remaining)
    
    # Verify 4AP-free
    A_final = set(best_A)
    for d in range(1, (N - 1) // 3 + 1):
        for x in range(1, N - 3 * d + 1):
            if all((x + j * d) in A_final for j in range(4)):
                # Remove one element to break the 4AP
                A_final.discard(x + 3 * d)
    
    # Ensure nonempty
    if not A_final:
        A_final = {1}
    
    return sorted(A_final)
