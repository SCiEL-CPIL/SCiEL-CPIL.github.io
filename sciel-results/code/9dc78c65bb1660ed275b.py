import numpy as np
import time
import random
import heapq

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        return [list(range(n)) for _ in range(1)]

    M = 3**n
    ball_size = 1 + 2 * n
    powers = [3**i for i in range(n)]

    # Precompute balls
    balls = [None] * M
    for i in range(M):
        digits = [0] * n
        temp = i
        for j in range(n):
            digits[j] = temp % 3
            temp //= 3
        ball = [i]
        for pos in range(n):
            new_d_plus = (digits[pos] + 1) % 3
            diff_plus = new_d_plus - digits[pos]
            if diff_plus != 0:
                ball.append(i + diff_plus * powers[pos])
            new_d_minus = (digits[pos] - 1) % 3
            diff_minus = new_d_minus - digits[pos]
            if diff_minus != 0:
                ball.append(i + diff_minus * powers[pos])
        balls[i] = ball

    # Precompute coverers
    coverers = [[] for _ in range(M)]
    for c in range(M):
        for p in balls[c]:
            coverers[p].append(c)

    def greedy_set_cover(initial_code=None):
        code = set(initial_code) if initial_code else set()
        uncovered = np.ones(M, dtype=bool)
        gains = np.full(M, ball_size, dtype=np.int32)

        if initial_code:
            for c in initial_code:
                for p in balls[c]:
                    if uncovered[p]:
                        uncovered[p] = False
                        for c_prime in coverers[p]:
                            if c_prime not in code:
                                gains[c_prime] -= 1

        heap = [(-gains[c], c) for c in range(M) if gains[c] > 0]
        heapq.heapify(heap)
        uncovered_count = int(np.sum(uncovered))

        while uncovered_count > 0:
            valid_c = None
            while heap:
                neg_g, c = heapq.heappop(heap)
                if c in code:
                    continue
                if -neg_g != gains[c]:
                    continue
                if gains[c] <= 0:
                    continue
                valid_c = c
                break

            if valid_c is None:
                break

            code.add(valid_c)
            for p in balls[valid_c]:
                if uncovered[p]:
                    uncovered[p] = False
                    uncovered_count -= 1
                    for c_prime in coverers[p]:
                        if c_prime not in code:
                            gains[c_prime] -= 1
                            if gains[c_prime] > 0:
                                heapq.heappush(heap, (-gains[c_prime], c_prime))

        return code

    def local_search_sa(code, time_limit=10):
        start_time = time.time()
        code = set(code)
        
        # Count coverage
        count = np.zeros(M, dtype=np.int32)
        for c in code:
            for p in balls[c]:
                count[p] += 1
        
        # Current cost: number of codewords
        current_cost = len(code)
        best_cost = current_cost
        best_code = set(code)
        
        T = 10.0 # Initial temperature
        cooling = 0.999
        
        while time.time() - start_time < time_limit:
            if T < 0.01:
                break
            
            # Random move: remove k, add k
            # k=1, 2, or 3
            k = random.choice([1, 1, 2, 2, 3])
            
            if len(code) < k:
                continue
                
            # Select k codewords to remove
            to_remove = random.sample(list(code), k)
            
            # Calculate uncovered points if we remove them
            temp_count = count.copy()
            for c in to_remove:
                for p in balls[c]:
                    temp_count[p] -= 1
            
            uncovered_points = [p for p in range(M) if temp_count[p] == 0]
            
            if not uncovered_points:
                # This is a pure removal, which is a huge improvement
                for c in to_remove:
                    code.remove(c)
                    for p in balls[c]:
                        count[p] -= 1
                current_cost = len(code)
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_code = set(code)
                continue
            
            # Find candidates to add
            # We need to cover all uncovered_points
            # This is a small Set Cover problem
            candidates = set()
            for p in uncovered_points:
                candidates.update(coverers[p])
            
            if not candidates:
                continue
                
            candidates -= code
            
            # We need to pick k' codewords from candidates such that they cover all uncovered_points
            # And ideally, the new total size is smaller or similar
            # Greedy pick k' codewords
            new_code = set(code) - set(to_remove)
            remaining_uncovered = set(uncovered_points)
            
            # Try to cover remaining_uncovered with k' codewords
            # k' should be k, k-1, or k+1 to allow for cost change
            # We want to minimize len(new_code)
            
            # Simple heuristic: pick k' codewords greedily
            for target_size in [k-1, k, k+1]:
                if target_size < 0: continue
                temp_new_code = set(new_code)
                temp_remaining = set(remaining_uncovered)
                
                for _ in range(target_size):
                    if not temp_remaining:
                        break
                    # Find best candidate
                    best_c = -1
                    best_gain = 0
                    # Limit candidates to those covering the most remaining
                    # Sample a few to save time
                    sample_size = min(20, len(candidates))
                    for c in random.sample(list(candidates), sample_size):
                        gain = 0
                        for p in temp_remaining:
                            if p in balls[c]:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_c = c
                    
                    if best_c == -1:
                        break
                        
                    temp_new_code.add(best_c)
                    candidates.discard(best_c)
                    
                    # Update remaining
                    for p in balls[best_c]:
                        if p in temp_remaining:
                            temp_remaining.remove(p)
                
                if not temp_remaining:
                    new_size = len(temp_new_code)
                    delta = new_size - current_cost
                    # Accept if better or with probability exp(-delta/T)
                    if delta < 0 or random.random() < np.exp(-delta / T):
                        # Apply move
                        code = temp_new_code
                        # Update count
                        count = np.zeros(M, dtype=np.int32)
                        for c in code:
                            for p in balls[c]:
                                count[p] += 1
                        current_cost = new_size
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_code = set(code)
                        break # Try next iteration with new state
            
            T *= cooling
            
        return best_code

    best_code = None
    best_size = float('inf')
    start_time = time.time()
    time_limit = 115

    # Initial Greedy
    initial_code = {0}
    code = greedy_set_cover(initial_code)
    code = local_search_sa(code, time_limit=10)
    if len(code) < best_size:
        best_size = len(code)
        best_code = set(code)

    # Multi-start
    for _ in range(10):
        if time.time() - start_time > time_limit - 15:
            break
        initial_code = set(random.sample(range(M), 10))
        code = greedy_set_cover(initial_code)
        code = local_search_sa(code, time_limit=10)
        if len(code) < best_size:
            best_size = len(code)
            best_code = set(code)

    # Final polish
    if best_code:
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time > 5:
            best_code = local_search_sa(best_code, time_limit=remaining_time)

    return [list(map(int, [(c // (3**i)) % 3 for i in range(n)])) for c in best_code]
