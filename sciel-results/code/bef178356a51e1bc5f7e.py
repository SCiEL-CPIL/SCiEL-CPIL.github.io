import random
import time
from itertools import product

def construct(k, g):
    # Target: 3-regular graph, girth g.
    # We assume k=3, g=16 based on the task description.
    # But the interface passes k and g.
    # If k != 3 or g != 16, we might need a different approach, 
    # but the task says "evaluator passes this task's degree k and girth g".
    # The seed program handles general k, g via LUW.
    # However, the task is specifically about n(3,16).
    # I will assume k=3 and g=16 for the optimized path, 
    # and fallback to LUW for other cases to ensure validity.
    
    if k != 3 or g != 16:
        # Fallback to a simple valid construction for other cases
        # or just use the LUW logic if needed. 
        # For now, return a trivial valid graph if possible or raise.
        # Given the context, it's likely always 3, 16.
        pass

    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Moore bound for bipartite 3-regular graph of girth 16 (2*8)
    # n >= 2 * (2^8 - 1) = 510
    # n = 2N, so 2N >= 510 => N >= 255
    # We search for N starting from 256.
    
    min_N = 256
    max_N = 2000  # Upper limit for search
    
    # To speed up search, we can use randomization
    random.seed(42)
    
    best_N = None
    best_A = None
    
    for N in range(min_N, max_N):
        if time.time() > deadline:
            break
            
        # Try random sets A of size 3
        # We need to avoid a + b = 0 mod N
        # We can just generate and check
        
        # Optimization: We can pre-filter A to not contain pairs summing to 0
        # But checking is fast.
        
        # Number of attempts per N
        attempts = 5000
        if N > 300:
            attempts = 2000
        if N > 500:
            attempts = 1000
            
        for _ in range(attempts):
            if time.time() > deadline:
                break
                
            # Generate random A
            # Ensure distinct
            a1 = random.randint(1, N - 1)
            a2 = random.randint(1, N - 1)
            a3 = random.randint(1, N - 1)
            
            if a1 == a2 or a1 == a3 or a2 == a3:
                continue
            
            # Quick check for 4-cycles: a_i + a_j == 0
            # a1 + a2 == 0, a1 + a3 == 0, a2 + a3 == 0
            if (a1 + a2) % N == 0 or (a1 + a3) % N == 0 or (a2 + a3) % N == 0:
                continue
            
            A = (a1, a2, a3)
            
            # Check for cycles of length 2*k for k=3..7 (length 6..14)
            # A cycle of length 2k exists if there is a sequence a_i1...a_ik 
            # such that sum = 0 mod N and partial sums are distinct.
            
            valid = True
            for k_len in range(3, 8): # k is half the cycle length
                # Iterate all sequences of length k_len from A
                # 3^k_len possibilities
                # For k_len=7, 2187.
                # We can use recursion or product
                for seq in product(A, repeat=k_len):
                    # Check sum
                    s = 0
                    partials = set()
                    # 0 is the start
                    # We check if any partial sum repeats or if final sum is 0
                    # Note: The cycle closes if sum == 0.
                    # The vertices are 0, s1, s1+s2, ...
                    # If any partial sum is 0 before the end, it's a shorter cycle.
                    # If any partial sum repeats, it's a shorter cycle.
                    
                    # Optimization: early exit
                    for a in seq:
                        s = (s + a) % N
                        if s == 0:
                            # If this is the last element, it's a valid cycle candidate
                            # But we are checking for cycles SHORTER than 16.
                            # So if we hit 0 at step m < k_len, it's a shorter cycle.
                            # If we hit 0 at step k_len, it's a cycle of length 2*k_len.
                            # We want to AVOID cycles of length < 16.
                            # So if k_len < 8 and sum becomes 0, it's a bad cycle.
                            # Wait, if sum becomes 0 at step m < k_len, the cycle length is 2m.
                            # If m < 8, length < 16. Bad.
                            # If sum becomes 0 at step k_len, length is 2*k_len.
                            # If k_len < 8, length < 16. Bad.
                            # So ANY zero sum for k_len < 8 is bad?
                            # Not exactly. The partial sums must be distinct.
                            # If partial sum becomes 0 at step m, the vertices visited are 0, ..., 0.
                            # This is a cycle of length 2m.
                            # If 2m < 16, it's a short cycle.
                            # So we must ensure NO sequence of length m < 8 sums to 0.
                            # This implies we don't even need to check distinctness for the "hit 0" case, 
                            # because hitting 0 implies a cycle.
                            # But what if partial sums repeat without hitting 0?
                            # e.g. s1 = s2. Then s2 - s1 = 0 => a2 = 0? No, a2 is from A, non-zero.
                            # s1 = s3 => a2+a3 = 0. This is a 4-cycle.
                            # So checking for zero sums of subsequences is sufficient?
                            # A cycle of length 2k corresponds to a zero-sum sequence of length k.
                            # If a sequence of length k sums to 0, does it always form a cycle?
                            # Yes, if the partial sums are distinct.
                            # If partial sums are NOT distinct, say s_i = s_j with i < j,
                            # then sum(a_{i+1}...a_j) = 0. This is a shorter zero-sum sequence.
                            # So if there is ANY zero-sum sequence of length m < 8,
                            # then there is a cycle of length <= 2m < 16.
                            # Therefore, the condition "girth >= 16" is equivalent to:
                            # There is NO sequence of length m in {1, ..., 7} that sums to 0 mod N.
                            # (Length 1: a=0, impossible since A subset 1..N-1)
                            # (Length 2: a+b=0, already checked)
                            # (Length 3..7: need to check)
                            
                            # So we just need to check if any sequence of length m < 8 sums to 0.
                            pass
                            
                    # We can break early if we find a sum 0
                    # But we need to check ALL sequences of length m < 8.
                    # So we can just check for each m from 2 to 7 if there is a zero-sum sequence.
                    # We already checked m=2.
                    # So we check m=3..7.
                    
                    # To make this faster, we can restructure the check.
                    # Instead of iterating sequences and checking sum, 
                    # we can just iterate and if sum==0, mark invalid.
                    
                    if s == 0:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                best_N = N
                best_A = A
                break
        
        if best_N:
            break

    if best_N is None:
        # If no solution found, return a known valid graph or fallback
        # For now, return a large N that is likely valid?
        # Or use the LUW graph as fallback to ensure validity.
        # But LUW is slow to generate? No, it's fast.
        # Let's just return a random large N that passed a quick check?
        # Better to return a definitely valid graph.
        # If search fails, we might timeout.
        # Let's try to find a solution with a larger N if small ones fail.
        # But the loop above goes up to 2000.
        
        # If still None, we have a problem.
        # We can construct a specific known graph if we know one.
        # But we don't have one hardcoded.
        # Let's assume we find one.
        
        # Fallback: Return a trivial 3-regular graph? 
        # No, must have girth 16.
        # If we didn't find one, we return the best effort?
        # The evaluator expects a valid graph.
        # If we return an invalid graph, score is 0.
        # We must ensure we return a valid graph.
        
        # Let's use the LUW construction as a guaranteed fallback.
        return _luw_fallback(k, g)

    N = best_N
    A = best_A
    
    n = 2 * N
    edges = []
    
    # Build edges
    # Left vertices 0..N-1, Right vertices N..2N-1
    for i in range(N):
        for a in A:
            j = (i + a) % N
            edges.append((i, N + j))
            
    return n, edges

def _luw_fallback(k, g):
    # Copy of the original logic to ensure a valid graph is returned
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
