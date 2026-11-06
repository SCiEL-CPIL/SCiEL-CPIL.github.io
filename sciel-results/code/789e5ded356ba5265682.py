"""
Strategy: Dynamic Priority Learning via Internal Optimization
Instead of a static heuristic, we use the compute budget to:
1. Run multiple instances of a greedy+local-search algorithm to find large cap sets.
2. Learn a priority score for each vector based on its frequency in these optimal sets.
3. Return these learned priorities in the final `priority` function.
"""

import time
import random
import sys
from collections import defaultdict

# Global state to store learned priorities
_LEARNED_PRIORITIES = {}
_LEARNING_COMPLETE = False
_START_TIME = time.time()

def _int_to_tuple(i, n):
    """Convert integer to tuple in Z_3^n."""
    res = []
    for _ in range(n):
        res.append(i % 3)
        i //= 3
    return tuple(res)

def _tuple_to_int(v):
    """Convert tuple in Z_3^n to integer."""
    res = 0
    for i, x in enumerate(v):
        res += x * (3 ** i)
    return res

def _is_cap_set_incremental(existing_set, new_vec, n):
    """
    Check if adding new_vec to existing_set keeps it a cap set.
    existing_set is a list of tuples.
    """
    # For each pair in existing_set, check if (a, b, new_vec) forms an AP.
    # AP condition: a + b + c = 0 mod 3.
    # So for any a, b in existing_set, check if (new_vec + a + b) == 0 mod 3.
    # Note: We don't need to check pairs involving new_vec and itself (distinct vectors).
    
    for i in range(len(existing_set)):
        a = existing_set[i]
        for j in range(i + 1, len(existing_set)):
            b = existing_set[j]
            # Check if a, b, new_vec form an AP
            # a + b + new_vec = 0 mod 3
            is_ap = True
            for k in range(n):
                if (a[k] + b[k] + new_vec[k]) % 3 != 0:
                    is_ap = False
                    break
            if is_ap:
                return False
    return True

def _greedy_with_local_search(n, time_limit, seed=None):
    """
    Run greedy construction with local search improvement.
    Returns the best cap set found.
    """
    if seed is not None:
        random.seed(seed)
    
    start_time = time.time()
    
    # Generate all vectors
    total_vectors = 3 ** n
    all_vectors = [_int_to_tuple(i, n) for i in range(total_vectors)]
    
    # Shuffle to break symmetry
    random.shuffle(all_vectors)
    
    cap_set = []
    cap_set_set = set()
    
    # Phase 1: Greedy Insertion
    for v in all_vectors:
        if time.time() - start_time > time_limit * 0.5:
            break
            
        # Check if v can be added
        if _is_cap_set_incremental(cap_set, v, n):
            cap_set.append(v)
            cap_set_set.add(v)
            
    # Phase 2: Local Search (Swap Improvement)
    # Try to swap an element in cap_set with an element outside to increase size?
    # Actually, swapping keeps size same. We want to INCREASE size.
    # To increase size, we need to find a vector outside that can be added.
    # But if greedy failed to add it, it conflicts with some pair.
    # Maybe removing one conflicting vector allows adding two new ones?
    # This is a "2-move" or "k-move" local search.
    
    # Simpler Local Search: Try to replace a "bad" vector with a "good" one?
    # Since we want max size, let's try:
    # 1. Identify vectors that were rejected.
    # 2. For each rejected vector v, find the pairs (a,b) in cap_set that conflict with v.
    # 3. Try removing a or b and see if we can add v AND another vector.
    
    # Given time constraints, let's stick to a simpler heuristic improvement:
    # Random Restart Greedy with different shuffles is often better than complex local search for small n.
    # But let's implement a basic 1-move swap to improve "quality" (spread) which might help greedy in future runs?
    # Actually, for the LEARNING phase, we just need to find large sets.
    
    # Let's do a few random restarts of the greedy if time permits.
    best_size = len(cap_set)
    best_set = list(cap_set)
    
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 1.0:
        # Try a few more random shuffles
        restarts = 0
        while time.time() - start_time < time_limit * 0.8 and restarts < 5:
            random.shuffle(all_vectors)
            curr_set = []
            curr_set_set = set()
            
            for v in all_vectors:
                if time.time() - start_time > time_limit * 0.9:
                    break
                if _is_cap_set_incremental(curr_set, v, n):
                    curr_set.append(v)
                    curr_set_set.add(v)
            
            if len(curr_set) > best_size:
                best_size = len(curr_set)
                best_set = list(curr_set)
            restarts += 1
            
    return best_set

def _learn_priorities(n, time_budget):
    """
    Run multiple optimization trials to learn which vectors are "good".
    """
    global _LEARNED_PRIORITIES, _LEARNING_COMPLETE
    
    if _LEARNING_COMPLETE:
        return
        
    start = time.time()
    # We want to spend most of the budget here, leaving ~5s for the final priority calls if needed?
    # Actually, priority() is called by the evaluator. The evaluator calls it for EVERY vector.
    # If n is large, 3^n is huge. The evaluator will call priority 3^n times.
    # If n is small (e.g., 5, 6, 7), 3^n is manageable (243, 729, 2187).
    # If n is large (e.g., 10, 20), 3^n is huge.
    # The task says "vectors of {0,1,2}^n".
    # Usually, these tasks have small n for exact search or medium n for heuristic.
    # If n is large, we can't store priorities for all vectors in a dict easily if we don't have a compact representation.
    # But we can compute a hash or use the integer representation as key.
    
    # Strategy:
    # 1. Determine if n is small enough for full enumeration in memory.
    # 2. Run several trials of _greedy_with_local_search.
    # 3. Count frequency of each vector in the "best" sets found.
    # 4. Assign priority based on frequency.
    
    max_n_for_full = 8 # 3^8 = 6561. Safe.
    
    if n <= max_n_for_full:
        # Run multiple trials
        trials = 0
        # Allocate 80% of time budget to learning
        learn_time_limit = time_budget * 0.8
        
        # We need to distribute time among trials.
        # Let's do at least 1 trial, and more if time allows.
        # Estimate time per trial.
        # For n=8, 3^8=6561. Greedy is O(N^2 * n) roughly. 6561^2 * 8 is too slow in Python.
        # _is_cap_set_incremental is O(|S|^2 * n). |S| grows to ~cap set size.
        # Max cap set size for n=8 is around 50-100?
        # 100^2 * 8 = 80000 ops per check. 6561 checks. ~500M ops. Too slow.
        
        # Optimization: Use bitsets or numpy?
        # Or just accept that for larger n, we can't do full enumeration in the learning phase.
        # If n is large, we fall back to a static heuristic.
        
        # Let's optimize _is_cap_set_incremental.
        # Precompute sums?
        
        # Revised Plan for Learning:
        # If n is small (<=6), do full learning.
        # If n is larger, use a faster heuristic for the priority function directly, without heavy learning.
        
        if n <= 6:
            # Full learning
            frequency_counts = defaultdict(int)
            
            # Run trials
            # Time per trial estimate: 3^n * (cap_size^2 * n)
            # Let's just run as many as we can within learn_time_limit.
            
            seed = 42
            while time.time() - start < learn_time_limit:
                best_set = _greedy_with_local_search(n, 1.0, seed=seed)
                for v in best_set:
                    frequency_counts[v] += 1
                seed += 1
                # Small sleep to avoid burning CPU if it's too fast? No, we want to use CPU.
                
            # Normalize frequencies to [0, 1]
            max_freq = max(frequency_counts.values()) if frequency_counts else 1
            for v, count in frequency_counts.items():
                _LEARNED_PRIORITIES[v] = count / max_freq
                
            # For vectors not in the set, assign low priority.
            # We don't need to iterate all vectors now, we can handle it in priority().
            
        else:
            # For larger n, learning is too slow.
            # We will use a smarter static heuristic in priority().
            pass
            
    _LEARNING_COMPLETE = True

def priority(vector, n):
    """
    Returns priority for a vector.
    The evaluator sorts vectors by descending priority.
    """
    global _LEARNING_COMPLETE, _START_TIME, _LEARNED_PRIORITIES
    
    # Check if we have finished learning.
    # If not, we need to initiate learning.
    # But priority() is called potentially many times.
    # We should only do learning once.
    
    # Estimate total time available.
    # The task says 300s limit.
    # We don't know exactly how much time is left, but we can track start time.
    # However, priority() is called sequentially.
    # The first call triggers the learning block if not done.
    
    if not _LEARNING_COMPLETE:
        # We need to estimate how much time we have.
        # Let's assume we have ~250s left for safety.
        # But wait, if n is large, learning is skipped.
        
        # Heuristic for time budget:
        # If n is small, we can spend time.
        # If n is large, we return immediately.
        
        # Let's set a hard limit for learning: 10 seconds for small n.
        # This is safe and won't kill the process.
        # If the evaluator calls priority for 3^n vectors, and n=6, 729 calls.
        # 10 seconds is plenty.
        
        # However, if n is very small, we can spend more.
        # Let's use a dynamic budget.
        # Actually, simpler: Just spend up to 5 seconds on learning.
        # This ensures we finish well within 300s.
        
        # But what if the evaluator expects us to use more time?
        # "The evaluator measures the raw value".
        # If we return a good priority function, the greedy algorithm will find a good set.
        # The greedy algorithm itself is run by the evaluator?
        # "vectors ... are added greedily in descending priority when they keep the set a cap set"
        # Yes, the evaluator runs the greedy algorithm.
        # So my priority function just needs to be GOOD.
        # It doesn't need to run the greedy algorithm itself to score, but I can use my time to figure out a good priority.
        
        # So, spending 5-10 seconds to learn a good priority map is a great trade-off.
        
        _learn_priorities(n, time_budget=10.0)
        
    # Now, return the priority.
    
    if n <= 6:
        if vector in _LEARNED_PRIORITIES:
            return _LEARNED_PRIORITIES[vector]
        else:
            # Vector was never in a good set. Give it low priority.
            # But we need a deterministic tie-breaker for vectors with same (low) priority.
            # Use a hash of the vector.
            h = hash(vector) % 1000
            return 0.0 + h * 1e-6
    else:
        # For larger n, we didn't learn. Use a static heuristic.
        # Which static heuristic is best?
        # 1. Sparsity (low weight)
        # 2. Balanced (weight ~ n/3)
        # 3. "Autocorrelation" low
        
        # Let's try a mix:
        # Prefer vectors with weight close to n/3?
        # Or just sparse?
        
        # Research suggests that for cap sets, "structured" vectors are good.
        # Let's try: priority = - (sum of squares) ?
        # Or just return a value based on weight.
        
        weight = sum(1 for x in vector if x != 0)
        
        # Heuristic: Vectors with weight 1 or 2 are good building blocks.
        # Vectors with weight n/3 might be good for filling.
        
        # Let's use a Gaussian-like preference for weight around n/3?
        # Or simply: lower weight is better?
        
        # Let's try: Priority = -weight.
        # This is what the current program does (sort of).
        
        # To improve, let's add a small random-looking deterministic noise to break ties?
        # No, noise can hurt if the greedy algorithm is sensitive.
        
        # Let's try a different static heuristic for large n:
        # Map vector to integer, take modulo some number? No.
        
        # Let's use the "balanced" heuristic:
        # Score = -abs(weight - n/3.0)
        
        score = -abs(weight - n / 3.0)
        
        # Add a tiny deterministic tie-breaker based on the vector content
        # to ensure consistent ordering.
        tie_breaker = 0
        for i, x in enumerate(vector):
            tie_breaker += x * (i + 1)
            
        return score + tie_breaker * 1e-9

# EVOLVE-BLOCK-END
