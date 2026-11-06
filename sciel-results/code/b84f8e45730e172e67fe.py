"""Narrow admissible tuple: fixed-diameter maximization in [0, N]."""
import time
import math
import random
import heapq

N = 5000
DEADLINE = 315.0  # safety margin before 330s kill

def _primes(limit):
    """Sieve of Eratosthenes."""
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            sieve[p * p::p] = bytearray(len(sieve[p * p::p]))
    return [p for p in range(2, limit + 1) if sieve[p]]

def construct():
    start_time = time.time()
    
    # Get primes up to N+1 (max possible tuple size is N+1)
    all_primes = _primes(N + 1)
    prime_set = set(all_primes)
    
    # Precompute residue lists for blocking logic
    # residue_lists[p][r] = list of v in [0, N] such that v % p == r
    residue_lists = {}
    for p in all_primes:
        lists = [[] for _ in range(p)]
        for v in range(N + 1):
            lists[v % p].append(v)
        residue_lists[p] = lists

    # Phase 1: Greedy removal from full set
    points = list(range(N + 1))
    
    while True:
        if time.time() - start_time > DEADLINE:
            break
        
        m = len(points)
        if m == 0:
            break
        
        primes_to_check = [p for p in all_primes if p <= m]
        
        changed = False
        for p in primes_to_check:
            counts = [0] * p
            for v in points:
                counts[v % p] += 1
            
            if min(counts) > 0:
                min_count = min(counts)
                min_residues = [r for r in range(p) if counts[r] == min_count]
                target_residue = min_residues[0]
                new_points = [v for v in points if v % p != target_residue]
                if len(new_points) < len(points):
                    points = new_points
                    changed = True
                    break
        
        if not changed:
            break
    
    # Phase 2: Optimized Add-back with Block Count Tracking
    best_points = list(points)
    best_len = len(points)
    
    def try_add_back_optimized(initial_points, time_limit):
        current = list(initial_points)
        current_set = set(current)
        m = len(current)
        
        # Initialize state
        # res_counts[p] = [counts for each residue]
        # missing_count[p]
        # critical_residue[p] = -1 or the unique missing residue
        # block_count[v] = number of primes blocking v
        
        res_counts = {}
        missing_count = {}
        critical_residue = {}
        
        # Initialize counts
        for p in all_primes:
            rc = [0] * p
            for v in current:
                rc[v % p] += 1
            mc = sum(1 for r in rc if r == 0)
            res_counts[p] = rc
            missing_count[p] = mc
            if mc == 1:
                for r in range(p):
                    if rc[r] == 0:
                        critical_residue[p] = r
                        break
            else:
                critical_residue[p] = -1
        
        # Initialize block_count
        block_count = [0] * (N + 1)
        for p in all_primes:
            cr = critical_residue[p]
            if cr != -1:
                # Increment block_count for all v in residue_lists[p][cr]
                for v in residue_lists[p][cr]:
                    block_count[v] += 1
        
        # Initialize safe candidates heap
        # We want to pick any safe candidate. To make it "random" or diverse,
        # we can just pick from a set.
        safe_candidates = set()
        for v in range(N + 1):
            if v not in current_set and block_count[v] == 0:
                safe_candidates.add(v)
        
        added = 0
        while safe_candidates:
            if time.time() > time_limit:
                break
            
            # Pick a candidate. To introduce randomness, we can pop a random one.
            # Or we can use a heap if we want "degree" based.
            # For now, random selection from safe set.
            v = random.choice(list(safe_candidates))
            safe_candidates.remove(v)
            
            # Check if v is still safe? It should be, as block_count didn't change for it
            # unless we added something that changed criticality.
            # But we remove from safe_candidates when we add.
            # However, adding v might change criticality of primes, which changes block_counts.
            
            # Add v
            current.append(v)
            current_set.add(v)
            m += 1
            added += 1
            
            # Update state for all primes
            for p in all_primes:
                r = v % p
                old_count = res_counts[p][r]
                old_mc = missing_count[p]
                old_cr = critical_residue[p]
                
                res_counts[p][r] = old_count + 1
                
                if old_count == 0:
                    new_mc = old_mc - 1
                else:
                    new_mc = old_mc
                
                missing_count[p] = new_mc
                
                # Determine new critical residue
                if new_mc == 1:
                    # Find the unique missing residue
                    new_cr = -1
                    for rr in range(p):
                        if res_counts[p][rr] == 0:
                            new_cr = rr
                            break
                else:
                    new_cr = -1
                
                critical_residue[p] = new_cr
                
                # Update block_count and safe_candidates
                if old_cr != new_cr:
                    # Unblock old critical residue
                    if old_cr != -1:
                        for u in residue_lists[p][old_cr]:
                            if u not in current_set:
                                block_count[u] -= 1
                                if block_count[u] == 0:
                                    safe_candidates.add(u)
                    
                    # Block new critical residue
                    if new_cr != -1:
                        for u in residue_lists[p][new_cr]:
                            if u not in current_set:
                                if block_count[u] == 0:
                                    safe_candidates.discard(u)
                                block_count[u] += 1
        
        return current

    # Try multiple random restarts
    # The "randomness" comes from the order in which we pick from safe_candidates
    # But since we rebuild from scratch each time, we need to vary the initial state?
    # No, the initial state is the result of Phase 1.
    # The randomness is in the choice of v from safe_candidates.
    
    # To get different results, we can shuffle the initial points? No, that doesn't change the set.
    # The set is fixed. The order of addition matters because it affects which primes become critical.
    # So different random choices of v will lead to different final sets.
    
    num_trials = 500
    for trial in range(num_trials):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        current = try_add_back_optimized(best_points, DEADLINE - 5)
        
        if len(current) > best_len:
            best_points = current
            best_len = len(current)
            
    return best_points
