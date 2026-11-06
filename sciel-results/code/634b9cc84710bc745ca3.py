import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    def time_left():
        return deadline - time.time()

    # Primes 1 mod 4 for M
    # We need enough primes to construct diverse M values
    p_1mod4 = [5, 13, 17, 29, 37, 41, 53, 61, 73, 89, 97, 101, 109, 113, 137, 149, 157, 173, 181, 193, 197, 229, 233, 241, 257, 269, 277, 281, 293, 313, 317, 337, 349, 353, 373, 389, 397, 401, 409, 421, 433, 449, 457, 461, 509, 521, 541, 557, 569, 577, 593, 599, 601, 613, 617, 641, 653, 661, 673, 677, 701, 709, 713, 733, 757, 761, 769, 773, 797, 809, 811, 821, 829, 853, 857, 877, 881, 929, 937, 941, 953, 977, 991, 997]

    # Primes for Q (primorial base)
    primes_all = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61]

    best_N = 0
    best_pairs = []
    best_score_val = 0

    # Try different Q values
    # Q = (product of first k primes)^2
    # We want to balance M size.
    # Larger k -> Smaller M -> Faster enumeration, but potentially fewer pairs.
    # Smaller k -> Larger M -> Slower enumeration, but potentially more pairs.
    
    # We'll try k from 1 to 10.
    # For each k, we find multiple candidate Ms.
    # Then we enumerate pairs for each candidate.
    
    for k in range(1, 11):
        if time_left() < 20:
            break
            
        S = 1
        for i in range(k):
            S *= primes_all[i]
        Q = S * S
        
        M_max = 10**18 // Q
        if M_max < 2:
            continue
        
        # Find top-K candidate M <= M_max
        # M = prod p_i^e_i, p_i in p_1mod4
        # Heuristic: Maximize prod (e_i + 1) is a good proxy for r_2(M) = 4 * prod (e_i + 1)
        # But we want to explore the space.
        
        # We will use a DFS that keeps the top 50 candidates by score.
        # To ensure diversity, we might also want to consider slightly lower scores
        # but different structures? For now, top 50 by divisor count is a solid structural change
        # from "top 1".
        
        candidates = [] # List of (score, M)
        
        # Limit the depth of DFS to first 30 primes to keep it fast but broad
        search_primes = p_1mod4[:30]
        
        # DFS to find top candidates
        # We need to be careful with recursion depth and time.
        # Let's use a priority queue or just a list and sort at the end?
        # Given the branching factor, the number of nodes visited can be large.
        # But M_max limits depth.
        
        # A better approach for "Top K":
        # Use a max-heap of size K? Or just collect all valid M and sort?
        # "Collect all" might be too many if M_max is large.
        # Let's estimate.
        # If k=1, Q=4, M_max=2.5e17.
        # Primes: 5, 13, ...
        # 5^e <= 2.5e17 => e <= 24.
        # The number of such M is roughly the number of 5-smooth (in 1mod4 primes) numbers.
        # This is manageable?
        # Let's try to collect top 100 via a limited DFS.
        
        # Instead of full DFS which might explode, we can use a beam search or
        # just DFS with a cutoff on the number of candidates stored.
        
        # Let's use a simple DFS that updates a list of top candidates.
        # To avoid O(N log K) for every node, we can just append and sort at the end
        # if the total number of visited nodes is not huge.
        # Or, we can use a heap.
        
        import heapq
        
        top_k = 100
        # heap elements: (-score, M)
        heap = []
        
        def dfs(idx, current_M, current_score, prev_exp):
            nonlocal heap
            if time_left() < 15:
                return
            
            # Pruning: if current_score is 0, skip (shouldn't happen)
            # We want to keep top K scores.
            if len(heap) < top_k:
                heapq.heappush(heap, (-current_score, current_M))
            else:
                if current_score > -heap[0][0]:
                    heapq.heapreplace(heap, (-current_score, current_M))
            
            if idx == len(search_primes):
                return
            
            p = search_primes[idx]
            
            # Max exponent for this prime
            val = 1
            e = 0
            while current_M * val <= M_max:
                if e > prev_exp:
                    break
                next_M = current_M * val
                next_score = current_score * (e + 1)
                
                dfs(idx + 1, next_M, next_score, e)
                
                val *= p
                e += 1
                if val > M_max:
                    break

        dfs(0, 1, 1, 100)
        
        if not heap:
            continue
            
        # Extract candidates, sort by score descending
        # Note: heap contains (-score, M)
        # We want to try the highest scores first.
        # But we also want to try diverse ones?
        # Let's sort by score descending.
        
        # Get unique Ms?
        # The DFS generates unique factorizations, so Ms are unique.
        
        cand_list = [(-neg_s, m) for neg_s, m in heap]
        cand_list.sort(key=lambda x: -x[0])
        
        # Iterate through candidates
        for score, M in cand_list:
            if time_left() < 5:
                break
                
            N = Q * M
            if N < 2 or N > 10**18:
                continue
                
            # Enumerate pairs for this specific M
            # x^2 + y^2 = M
            # x <= y => x^2 <= M/2 => x <= sqrt(M/2)
            
            limit_x = int(math.isqrt(M // 2)) + 1
            
            # Check if enumeration is feasible
            # If limit_x is too large, we might skip.
            # But high score M usually has many small factors, meaning M is "smooth",
            # which doesn't necessarily mean M is small.
            # However, if M is huge, limit_x is huge.
            # We must guard against this.
            
            # Heuristic: if limit_x > 5*10^6, enumeration takes ~1s.
            # We have 100s budget.
            # Let's allow up to 2*10^7 iterations per candidate if we have time.
            # But we have 100 candidates.
            # We should be smarter.
            
            # If limit_x is very large, it's likely M is large.
            # But wait, if M is large, Q is small (k small).
            # For k=1, M_max is 2.5e17. limit_x ~ 10^8. Too slow.
            # We should skip candidates where limit_x is too large to enumerate quickly.
            # Or, we should prioritize candidates with smaller M?
            # No, we want high score.
            
            # Let's check the time cost.
            # 10^7 iterations in Python ~ 0.5-1s.
            # If we have 100 candidates, we can't do 10^7 for each.
            
            # Strategy:
            # 1. Sort candidates by Score / limit_x ? No.
            # 2. Just try them in order of Score.
            # 3. If limit_x > 2*10^6, skip?
            #    That would miss large M with high scores.
            #    But large M is hard to enumerate.
            
            # Let's set a dynamic limit based on remaining time.
            # If we have 100s left, we can do ~100 * 10^7 iterations? No, 10^9 is too much.
            # 100s / 100 candidates = 1s per candidate.
            # 1s allows ~10^7 iterations.
            
            # So, if limit_x > 10^7, we might skip or do a partial search?
            # Partial search doesn't give a certified lower bound unless we find all pairs.
            # We MUST find all pairs to certify.
            
            # Therefore, we can only certify pairs if we can enumerate the full range.
            # So we must skip M where limit_x is too large for the remaining time.
            
            # Estimated time per iteration: ~0.1 microsecond? No, Python is slower.
            # Let's say 0.5 us per iteration (optimistic).
            # 10^7 iterations = 5s.
            # So if limit_x > 10^7, it takes > 5s.
            
            # We have ~100s.
            # We can afford ~20 such large enumerations.
            # Or ~200 small ones.
            
            # Let's just try. If it's too slow, we break.
            
            # Optimization:
            # If we already have a best_pairs with count C,
            # and the theoretical max pairs for this M is Score (since r_2(M) = 4*Score, pairs = Score/2 roughly? No.)
            # Number of pairs (a,b) with a<=b, a+b=N, a,b powerful.
            # a = Q x^2, b = Q y^2.
            # x^2 + y^2 = M.
            # Number of solutions (x,y) with x,y >= 1, x <= y.
            # This is roughly r_2(M) / 4 if M is not a square, etc.
            # r_2(M) = 4 * prod (e_i+1).
            # So max pairs is roughly prod (e_i+1).
            # If Score (prod(e_i+1)) <= len(best_pairs), we can skip this M entirely!
            # This is a HUGE pruning opportunity.
            
            if score <= best_score_val:
                continue
                
            # Now we know this M *could* beat the current best.
            # Check enumeration feasibility.
            
            # If limit_x is too big, we might time out.
            # Let's estimate time.
            # If limit_x > 5*10^7, it's risky.
            # But if score is high, it's worth trying.
            
            # Let's just run it with a strict time check.
            
            pairs = []
            M_val = M
            Q_val = Q
            isqrt = math.isqrt
            
            # We need to find all x in [1, limit_x] such that M-x^2 is a square.
            # To speed this up, we can precompute squares?
            # No, M is large.
            
            # Just loop.
            # To avoid timeout, we check time frequently.
            
            # If we time out, we return empty pairs for this M, and keep previous best.
            
            # One optimization:
            # If we find that x^2 + y^2 = M, we add to pairs.
            # We must find ALL such pairs.
            
            # Let's iterate x.
            # If limit_x is huge, this is slow.
            
            # Is there a faster way?
            # Factorization of M is known (from DFS).
            # We can generate all representations of M as sum of two squares directly from the factorization!
            # This is MUCH faster than scanning x.
            
            # Algorithm for sum of two squares from factorization:
            # M = prod p_i^{e_i}.
            # Representations are generated by multiplying Gaussian integers.
            # This is a standard algorithm.
            
            # Let's implement this instead of scanning x!
            # This changes the strategy from "Scan and Check" to "Generate from Factorization".
            # This is structurally different and much faster.
            
            # We need the exponents for M.
            # We didn't store them in the heap.
            # We need to store the factorization or re-factor M.
            # Re-factor M is fast since M is composed of small primes from p_1mod4.
            
            # Let's re-factor M using p_1mod4.
            
            exponents = []
            temp_M = M
            for p in search_primes:
                if temp_M % p == 0:
                    e = 0
                    while temp_M % p == 0:
                        temp_M //= p
                        e += 1
                    exponents.append((p, e))
                else:
                    # If p^2 > temp_M and temp_M > 1, temp_M is prime?
                    # But M is constructed from search_primes.
                    # So temp_M should become 1.
                    pass
            
            if temp_M != 1:
                # M has a prime factor not in search_primes?
                # This shouldn't happen if DFS was correct.
                # But if it does, we can't use the fast generation.
                # Fallback to scanning? Or skip?
                # Let's skip to be safe, or fallback to scan if small.
                # Actually, if temp_M > 1, it means our search_primes list was insufficient
                # or there's a bug.
                # Given M <= 10^18 and search_primes up to 997,
                # any prime factor > 997 would have to be large.
                # If M is composed of primes > 997, it must be a product of few such primes.
                # But our DFS only used search_primes. So temp_M MUST be 1.
                
                # Wait, if M=1, exponents is empty.
                # If M>1, and no primes divide it, then M is prime?
                # But M is generated by DFS from search_primes.
                # So this case should not occur.
                
                # Let's just skip this candidate if factorization fails.
                continue

            # Generate representations of M as sum of two squares.
            # M = prod p_i^{e_i}.
            # For each prime p_i = a_i^2 + b_i^2, we have a Gaussian prime.
            # We need to find all pairs (x,y) such that x^2 + y^2 = M.
            
            # This is a known problem.
            # Number of representations is 4 * prod (e_i + 1).
            # We need to generate them.
            
            # We can use the identity:
            # (a^2 + b^2)(c^2 + d^2) = (ac - bd)^2 + (ad + bc)^2 = (ac + bd)^2 + (ad - bc)^2
            
            # We start with 1 = 1^2 + 0^2.
            # For each prime power p^e, we multiply the current representations by the representations of p^e.
            
            # Representations of p^e:
            # If p = a^2 + b^2, then p^e has e+1 representations (up to sign/order).
            # Specifically, (a+ib)^k * (a-ib)^{e-k} for k=0..e.
            
            # This is getting complex to implement correctly and efficiently in Python within the time limit
            # without a library.
            
            # Given the time pressure and the risk of bugs in Gaussian integer arithmetic,
            # and the fact that the current scanning approach scored 38.4,
            # maybe the scanning approach is sufficient if we prune better.
            
            # Let's stick to the scanning approach but with the PRUNING based on Score.
            # The pruning `if score <= best_score_val: continue` is the key structural improvement.
            # It prevents us from wasting time on candidates that cannot beat the current best.
            
            # So, we only scan M if its theoretical max pairs (score) is greater than current best.
            
            # Now, is scanning feasible?
            # If score > best_score_val, and we have found a good best_score_val early,
            # we will skip most candidates.
            
            # We should sort candidates by score DESCENDING.
            # So we try the most promising ones first.
            # Once we find a high score, we skip all lower ones.
            
            # This is a huge improvement.
            
            # Let's implement the scan with time guard.
            
            # If limit_x is too large, we might still time out.
            # But if score is high, it's worth the risk.
            
            # Let's add a check: if limit_x > 2*10^7, we might skip?
            # No, let's try.
            
            # To make scanning faster, we can use numpy?
            # No, standard loop is fine.
            
            # One more optimization:
            # We can break early if we have found enough pairs?
            # No, we need ALL pairs to certify the count.
            
            # So we must scan the whole range.
            
            # Let's go.
            
            # If time runs out during scan, we discard this candidate's pairs.
            
            # Start scan
            found_pairs = []
            
            # To speed up, we can check time every 10000 iterations.
            
            # Also, we can use a set to avoid duplicates?
            # x <= y ensures uniqueness for (x,y) pairs.
            # a = Q x^2, b = Q y^2.
            # Since x <= y, a <= b.
            # Pairs are distinct.
            
            # Let's run the scan.
            
            # If limit_x is huge, this will be slow.
            # But hopefully, the pruning will save us.
            
            # Let's add a hard limit on limit_x for safety?
            # If limit_x > 5*10^7, it takes ~25s.
            # If we have 100s, we can afford 4 such scans.
            
            # Let's just run it.
            
            # Optimization: Use local variables for speed.
            M_local = M_val
            Q_local = Q_val
            isqrt_local = isqrt
            pairs_local = found_pairs
            
            # We need to know when to stop.
            # If time_left() < 5, break and discard.
            
            # Let's iterate.
            
            # If limit_x is very large, we might want to skip this candidate
            # if we have already found a good best.
            # But we are already in the "score > best_score" branch.
            
            # Let's just try.
            
            # To avoid infinite loop if time check is slow, we check time.
            
            # Note: If M is a square, x=y is possible.
            # If M is not a square, x != y.
            
            # Let's run.
            
            # If this takes too long, the whole program might time out.
            # We must ensure we return something.
            
            # Let's wrap the scan in a try/except or just rely on the time check.
            
            # If we break due to time, we don't update best_pairs.
            
            # Let's go.
            
            # One detail: x starts from 1.
            # What if x=0 is allowed?
            # "1 counts as powerful". 0 does not.
            # So x >= 1.
            
            # Let's scan.
            
            # If limit_x is > 10^7, we might want to skip if we have < 10s left?
            # No, we check time inside.
            
            # Let's do it.
            
            # To make it faster, we can step by 1.
            
            # If we find a pair, we add it.
            
            # Let's run the loop.
            
            # If the loop is too slow, we will time out.
            # But the pruning should help.
            
            # Let's implement.
            
            # Wait, I should check if `score` is indeed the max number of pairs.
            # r_2(M) = 4 * prod (e_i+1).
            # Number of pairs (x,y) with x,y >= 1, x <= y, x^2+y^2=M.
            # This is roughly r_2(M) / 4 if M is not a square.
            # If M is a square, it's (r_2(M) + 4) / 8 ?
            # Anyway, score = prod(e_i+1) is an upper bound on the number of pairs.
            # So `if score <= best_score_val: continue` is valid.
            
            # Let's run.
            
            # To be safe against timeout, I will add a check:
            # If limit_x > 2*10^7 and time_left() < 20, skip.
            
            if limit_x > 2 * 10**7 and time_left() < 20:
                continue
                
            # Scan
            for x in range(1, limit_x):
                if x % 1000000 == 0 and time_left() < 5:
                    break
                    
                x2 = x * x
                y2 = M_local - x2
                if y2 < 0:
                    break
                y = isqrt_local(y2)
                if y * y == y2:
                    if x <= y:
                        a = Q_local * x2
                        b = Q_local * y2
                        pairs_local.append([a, b])
                        
            if len(pairs_local) > best_score_val:
                best_pairs = pairs_local
                best_N = N
                best_score_val = len(pairs_local)

    # Fallback
    if not best_pairs:
        best_N = 5
        best_pairs = [[1, 4]]
        
    return {'N': best_N, 'pairs': best_pairs}

if __name__ == "__main__":
    res = construct()
    print(res['N'])
    print(len(res['pairs']))
