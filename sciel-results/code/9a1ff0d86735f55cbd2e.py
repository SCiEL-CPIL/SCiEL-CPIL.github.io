import time
import math
import random
from collections import Counter

N = 5000
DEADLINE = 320.0

def _sieve_primes(limit):
    if limit < 2:
        return []
    sieve = bytearray([1]) * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = bytearray(len(sieve[i * i::i]))
    return [i for i in range(2, limit + 1) if sieve[i]]

def construct():
    start_time = time.time()
    primes = _sieve_primes(N)
    P = len(primes)
    
    # Precompute range lists for each prime for faster iteration
    # range_lists[i] is the list of x in [0, N] such that x % p_i == r
    # We need this for all residues? No, we need it for the specific residue.
    # Let's precompute the base list of indices for each prime.
    # idx_lists[i] = list of x in [0, N] where x % p_i == 0.
    # Then for residue r, the indices are (x + r) % p_i? No, just shift.
    # Actually, storing the full list for each prime is memory heavy if we store all residues.
    # But we only need to iterate over the specific residue class.
    # range(r, N+1, p) is efficient in Python.
    
    # State:
    # r[i]: the missed residue for prime i
    # block_count[x]: number of primes p such that x % p == r[p]
    
    r = [0] * P
    block_count = [0] * (N + 1)
    
    # Initialize
    for i, p in enumerate(primes):
        for x in range(0, N + 1, p):
            block_count[x] += 1
            
    # surv_count[i][res] = number of x in [0, N] such that x % p_i == res and block_count[x] == 0
    # Note: x % p_i == res is the class we are considering for prime i.
    # If we change r[i] to res, then x in that class becomes unblocked by p_i.
    # The gain is: (number of x in class res that were blocked ONLY by p_i) - (number of x in class res that were unblocked by everyone else but now become blocked by p_i)
    # Wait, let's look at the delta in total survivors (S).
    # S = count(x where block_count[x] == 0)
    
    # Let's define:
    # zero_count[res] for prime i: count of x in class res (mod p_i) with block_count[x] == 0
    # one_count[res] for prime i: count of x in class res (mod p_i) with block_count[x] == 1
    # Note: The x counted in one_count for prime i must have block_count[x]==1.
    # Since x is in class r_old, and p_i is one of the blockers, if block_count[x]==1, it means p_i is the ONLY blocker for x.
    # So, if we move p_i away from r_old, these x become unblocked (block_count 1->0). Gain: +one_count[r_old].
    # If we move p_i to r_new, the x in r_new that currently have block_count[x]==0 will become blocked (0->1). Loss: -zero_count[r_new].
    # So Gain = one_count[r_old] - zero_count[r_new].
    
    # We need to maintain zero_count and one_count for all primes and all residues?
    # That is sum(p) ~ 10000 integers. Very small.
    # Let's use a 2D list: stats[i][res] = [zero_count, one_count]
    
    stats = [[0] * p for p in primes]
    
    # Initialize stats
    for x in range(N + 1):
        bc = block_count[x]
        if bc == 0:
            for i, p in enumerate(primes):
                res = x % p
                stats[i][res][0] += 1
        elif bc == 1:
            # Which prime is the blocker?
            # We need to find i such that x % p_i == r[i].
            # This is potentially expensive if done naively for every x.
            # But N=5000, P=669. 5000*669 ~ 3.3M ops. Acceptable for init.
            # Optimization: We can just check.
            for i, p in enumerate(primes):
                if x % p == r[i]:
                    stats[i][res][1] += 1
                    break # Only one prime can be the single blocker? 
                    # Wait, if block_count[x]==1, exactly one prime blocks it.
                    # So yes, break.
                    
    best_size = sum(1 for x in range(N + 1) if block_count[x] == 0)
    best_r = r[:]
    
    deadline = start_time + DEADLINE - 5.0
    
    # Local Search
    # We want to maximize best_size.
    # Move: change r[i] from old_r to new_r.
    # Gain = stats[i][old_r][1] - stats[i][new_r][0]
    
    # To make it faster, we can precompute the best candidate for each i?
    # best_new_r[i] = argmin(stats[i][res][0]) for res != r[i]
    # And best_gain[i] = stats[i][r[i]][1] - min(stats[i][res][0])
    
    # Let's do a loop.
    
    while time.time() < deadline:
        best_move = -1
        best_gain = 0
        best_new_r = 0
        best_old_r = 0
        
        # Find best move
        for i, p in enumerate(primes):
            old_r = r[i]
            # Find min zero_count for res != old_r
            min_zero = float('inf')
            min_res = -1
            
            # Optimization: If p is large, iterating all residues is expensive.
            # But sum(p) is small.
            for res in range(p):
                if res == old_r:
                    continue
                if stats[i][res][0] < min_zero:
                    min_zero = stats[i][res][0]
                    min_res = res
            
            if min_res == -1:
                continue
                
            one_old = stats[i][old_r][1]
            gain = one_old - min_zero
            
            if gain > best_gain:
                best_gain = gain
                best_move = i
                best_new_r = min_res
                best_old_r = old_r
        
        if best_move == -1:
            # No improving move. Try random perturbation.
            # Pick a random prime and a random new residue.
            if random.random() < 0.5:
                i = random.randint(0, P - 1)
                p = primes[i]
                old_r = r[i]
                new_r = random.randint(0, p - 1)
                while new_r == old_r:
                    new_r = random.randint(0, p - 1)
                best_move = i
                best_new_r = new_r
                best_old_r = old_r
            else:
                break
        
        # Apply move
        i = best_move
        p = primes[i]
        old_r = best_old_r
        new_r = best_new_r
        
        # Update block_count and stats
        # 1. Unblock old_r class for prime i
        # x in range(old_r, N+1, p)
        # For each x:
        #   block_count[x] -= 1
        #   If old_bc == 1: becomes 0.
        #       stats for prime i: res=old_r. zero_count[old_r] += 1?
        #       Wait, stats[i][res] counts x with x%p_i==res.
        #       For prime i, x is in old_r class.
        #       Before: block_count[x]==1. It contributed to stats[i][old_r][1].
        #       After: block_count[x]==0. It contributes to stats[i][old_r][0].
        #       So stats[i][old_r][1] -= 1, stats[i][old_r][0] += 1.
        #       Also, for other primes j, x might be in some class res_j.
        #       Before: block_count[x]==1. If x%p_j == r_j, it contributed to stats[j][res_j][1].
        #       After: block_count[x]==0. It contributes to stats[j][res_j][0].
        #       So for ALL j, stats[j][x%p_j][1] -= 1 and stats[j][x%p_j][0] += 1?
        #       NO. stats[j][res] counts x where x%p_j==res AND block_count[x]==k.
        #       If block_count[x] changes from 1 to 0, then for EVERY prime j,
        #       x is in class (x%p_j).
        #       So stats[j][x%p_j][1] should decrease by 1?
        #       Only if x was counted there. Yes, if block_count[x]==1, it was counted in stats[j][x%p_j][1] for all j?
        #       Wait. stats[j][res][1] counts x such that x%p_j==res AND block_count[x]==1.
        #       So yes, if block_count[x] was 1, it was in stats[j][x%p_j][1] for all j.
        #       Now it is 0, so it moves to stats[j][x%p_j][0].
        #       
        #   If old_bc == 2: becomes 1.
        #       Before: block_count[x]==2. Not in any stats[...][1] or [0].
        #       After: block_count[x]==1.
        #       It becomes counted in stats[j][x%p_j][1] for all j.
        #       So stats[j][x%p_j][1] += 1 for all j.
        #       
        #   If old_bc > 2: no change in stats (still >= 2).
        
        # 2. Block new_r class for prime i
        # x in range(new_r, N+1, p)
        # For each x:
        #   block_count[x] += 1
        #   If old_bc == 0: becomes 1.
        #       Before: block_count[x]==0. In stats[j][x%p_j][0] for all j.
        #       After: block_count[x]==1. In stats[j][x%p_j][1] for all j.
        #       So stats[j][x%p_j][0] -= 1, stats[j][x%p_j][1] += 1.
        #   If old_bc == 1: becomes 2.
        #       Before: in stats[j][x%p_j][1] for all j.
        #       After: in no stats.
        #       So stats[j][x%p_j][1] -= 1.
        #   If old_bc >= 2: no change in stats.
        
        # This update is O(P) per x.
        # Number of x affected: N/p_old + N/p_new.
        # Total cost per move: (N/p_old + N/p_new) * P.
        # P ~ 669. N ~ 5000.
        # For p=2: 2500 * 669 ~ 1.6M ops.
        # This might be too slow for many iterations.
        
        # Let's optimize the stats update.
        # Instead of updating all j, we can just recompute stats for the affected x?
        # Or, we can just recompute the "best move" search by scanning stats?
        # The stats update is the bottleneck.
        
        # Alternative: Don't maintain stats incrementally.
        # Just maintain block_count.
        # When searching for best move, calculate gain on the fly?
        # Gain(i, new_r) = count(x in new_r, bc==0) - count(x in old_r, bc==1)
        # This requires scanning N/p elements.
        # If we do this for all i and all new_r, it's sum(p) * N/p ~ N * P ~ 3.3M ops per iteration.
        # 3.3M ops is fast in Python (maybe 0.1s).
        # So we can do ~30 iterations in 3s? No, 3.3M ops is ~0.01s?
        # Let's check. 10^7 ops is ~0.1s in PyPy? In CPython ~1s?
        # 3.3M ops ~ 0.3s?
        # So we can do ~10 iterations in 3s.
        # 10 iterations is not enough for local search.
        
        # Let's stick to incremental stats update but optimize it.
        # We only need stats[i] for the prime i we are moving?
        # No, the gain for moving prime k depends on stats[k].
        # But the update of block_count affects stats for ALL primes.
        
        # Is there a way to avoid updating all stats?
        # We only need stats[i][res] for the "best move" calculation.
        # The best move calculation needs min(stats[i][res][0]) and stats[i][old_r][1].
        # So we need stats for all i.
        
        # Let's try to make the update faster.
        # Instead of a list of lists, use a flat array?
        # Or just accept the cost and see how many iterations we get.
        # Let's measure.
        
        # Update old_r unblock
        for x in range(old_r, N + 1, p):
            old_bc = block_count[x]
            block_count[x] = old_bc - 1
            new_bc = old_bc - 1
            
            if old_bc == 1:
                # 1 -> 0
                for j, q in enumerate(primes):
                    res_j = x % q
                    stats[j][res_j][1] -= 1
                    stats[j][res_j][0] += 1
            elif old_bc == 2:
                # 2 -> 1
                for j, q in enumerate(primes):
                    res_j = x % q
                    stats[j][res_j][1] += 1
            # else no change to stats
            
        # Update new_r block
        for x in range(new_r, N + 1, p):
            old_bc = block_count[x]
            block_count[x] = old_bc + 1
            new_bc = old_bc + 1
            
            if old_bc == 0:
                # 0 -> 1
                for j, q in enumerate(primes):
                    res_j = x % q
                    stats[j][res_j][0] -= 1
                    stats[j][res_j][1] += 1
            elif old_bc == 1:
                # 1 -> 2
                for j, q in enumerate(primes):
                    res_j = x % q
                    stats[j][res_j][1] -= 1
            # else no change to stats
            
        r[i] = new_r
        
        # Update best_size
        # We can track it incrementally?
        # Change in survivors:
        # x in old_r: 1->0 (gain 1), 2->1 (gain 0)
        # x in new_r: 0->1 (loss 1), 1->2 (loss 0)
        # So delta_size = (count of x in old_r with bc==1) - (count of x in new_r with bc==0)
        # This is exactly the gain we calculated!
        # Wait, did we calculate it correctly?
        # Gain = stats[i][old_r][1] - stats[i][new_r][0]
        # Yes.
        # So best_size += best_gain?
        # Only if best_gain > 0.
        # If we did a random move, we need to calculate the actual change.
        
        # Let's just recount. It's 5000 ops. Fast.
        current_size = sum(1 for x in range(N + 1) if block_count[x] == 0)
        if current_size > best_size:
            best_size = current_size
            best_r = r[:]
            
        # Check time
        if time.time() > deadline:
            break
            
    # Reconstruct result from best_r
    # We need to restore r to best_r and recalculate block_count?
    # Or we can just construct the set directly from best_r.
    # A number x is in the set if for all i, x % p_i != best_r[i].
    
    result = []
    for x in range(N + 1):
        keep = True
        for i, p in enumerate(primes):
            if x % p == best_r[i]:
                keep = False
                break
        if keep:
            result.append(x)
            
    return result
