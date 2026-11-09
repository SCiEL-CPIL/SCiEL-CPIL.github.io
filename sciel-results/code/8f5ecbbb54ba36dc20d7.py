import time
import random
import sys

# EVOLVE-BLOCK-START

def _is_valid_full(values):
    """Check if a full sequence has all distinct consecutive block sums."""
    n = len(values)
    if n == 0:
        return True
    seen = set()
    for i in range(n):
        total = 0
        for j in range(i, n):
            total += values[j]
            if total in seen:
                return False
            seen.add(total)
    return True

def _greedy_forward(values, max_val=256):
    """Greedy: pick smallest valid next element."""
    current = list(values)
    cur_sums = set()
    for i in range(len(current)):
        total = 0
        for j in range(i, len(current)):
            total += current[j]
            cur_sums.add(total)
    
    for x in range((current[-1] if current else 0) + 1, max_val + 1):
        tails = [x]
        total = x
        for a in reversed(current):
            total += a
            tails.append(total)
        if not cur_sums.intersection(tails):
            current.append(x)
            cur_sums.update(tails)
    return current

def _compute_sums(seq):
    """Compute all consecutive block sums for a sequence."""
    sums = set()
    n = len(seq)
    for i in range(n):
        total = 0
        for j in range(i, n):
            total += seq[j]
            sums.add(total)
    return sums

def _valid_extension(seq, x, cur_sums):
    """Check if x can be appended to seq without creating duplicate sums."""
    tails = [x]
    total = x
    for a in reversed(seq):
        total += a
        tails.append(total)
    return not cur_sums.intersection(tails)

def _sa_local_search(initial_seq, max_val=256, deadline=None, best_len=0):
    """
    Simulated Annealing to maximize sequence length.
    State: a valid strictly increasing sequence.
    Moves:
    1. Try to extend the sequence by adding a new element (if possible).
    2. If extension is not possible or to escape local minima, perform a 'shake':
       - Remove a random element from the end (or middle?) and try to re-insert it elsewhere or replace it.
       - Actually, a better move for increasing length is to try to 'swap' a suffix with a different valid suffix.
       
    Let's define the state as the sequence itself.
    Objective: Maximize length.
    
    Moves:
    1. Extension: Try to add a random valid x > last. If successful, length increases.
    2. Perturbation: If stuck, remove the last element (backtrack 1 step), then try a different next element.
       This is essentially a randomized DFS/Backtracking with a bias towards longer sequences.
    
    To make it SA-like:
    - Temperature T starts high, decreases.
    - We accept "worse" moves (shorter sequences) with probability exp(-delta/T) to escape local optima.
    - But since we want to maximize length, "worse" means shorter.
    - We can also try to "repair" a sequence that got stuck by swapping elements.
    
    A more robust approach for this specific constraint (distinct block sums):
    The space of valid sequences is a tree (each prefix has valid extensions).
    Standard DFS explores this tree.
    SA can be used to explore the tree more effectively by "jumping" between branches.
    
    However, implementing a full SA on a tree structure is tricky.
    Alternative: Use the SA idea to guide the choice of next element in a DFS, or to decide when to backtrack.
    
    Let's try a simpler "Iterated Local Search" (ILS) which is a form of SA:
    1. Start with a valid sequence (e.g., from greedy).
    2. Try to extend it.
    3. If stuck, perturb the sequence (e.g., remove last k elements, re-generate them randomly/greedily).
    4. Repeat.
    
    This is different from simple DFS because it explicitly "jumps" back and explores alternative branches with randomization, rather than just backtracking deterministically.
    """
    if deadline and time.time() > deadline:
        return initial_seq, best_len
    
    current = list(initial_seq)
    best_found = list(current)
    best_len_local = len(current)
    
    if not current:
        return current, best_len_local
    
    # We will use a time-limited loop to perform ILS steps
    rng = random.Random(12345)
    
    # Initial temperature or just a fixed number of iterations?
    # Time limit is better.
    
    while time.time() < deadline:
        # 1. Try to extend the current sequence greedily/randomly
        cur_sums = _compute_sums(current)
        last_val = current[-1]
        
        # Try to find a valid extension
        extended = False
        # Try random candidates first to diversify, then greedy?
        # Or just greedy to maximize length quickly, then perturb if stuck.
        
        # Let's try to extend as much as possible first (Greedy)
        # This ensures we are at a local maximum for this branch.
        extended_seq = _greedy_forward(current, max_val)
        
        if len(extended_seq) > best_len_local:
            best_len_local = len(extended_seq)
            best_found = list(extended_seq)
        
        current = extended_seq
        
        # Check if we are at a dead end (cannot extend further)
        cur_sums = _compute_sums(current)
        last_val = current[-1]
        can_extend = False
        for x in range(last_val + 1, max_val + 1):
            if _valid_extension(current, x, cur_sums):
                can_extend = True
                break
        
        if can_extend:
            # If we can still extend, the greedy loop above should have done it.
            # If it didn't, something is wrong. But _greedy_forward is exhaustive for smallest.
            # So if _greedy_forward returned, it's a local max (smallest valid).
            # We need to perturb to find a *larger* valid extension that might allow longer total length.
            pass
        
        # 2. Perturb: Backtrack a few steps and try a different path
        # How many steps to backtrack?
        # Randomly choose a backtrack depth between 1 and min(5, len(current)-1)
        # If len is small, backtrack less.
        
        if len(current) <= 1:
            # Can't perturb much, restart from scratch or different start?
            # For now, just break or try different start in outer loop.
            break
        
        backtrack_depth = rng.randint(1, min(5, len(current) - 1))
        prefix = current[:-backtrack_depth]
        
        # Now, from this prefix, try to generate a new suffix using a randomized greedy approach
        # Instead of always picking the smallest valid, pick a random valid one.
        # This explores different branches.
        
        new_suffix = []
        cur_sums = _compute_sums(prefix)
        last_val = prefix[-1] if prefix else 0
        
        # Try to build a suffix of length at least backtrack_depth (or more)
        # We don't have to match the old length, we just want to explore.
        
        # Randomized Greedy:
        # At each step, pick a random valid next element.
        # To bias towards longer sequences, we might prefer smaller elements?
        # Smaller elements leave more room for more elements.
        # So, sort valid candidates and pick one from the "front" with some randomness.
        
        current_suffix = []
        working_seq = list(prefix)
        working_sums = set(cur_sums)
        
        # Try to extend working_seq
        max_attempts = 100
        for _ in range(max_attempts):
            if time.time() > deadline:
                break
            last_val = working_seq[-1]
            candidates = []
            for x in range(last_val + 1, max_val + 1):
                if _valid_extension(working_seq, x, working_sums):
                    candidates.append(x)
            
            if not candidates:
                break
            
            # Pick a candidate. 
            # To escape local optima, we should sometimes pick larger candidates.
            # But larger candidates consume "sum budget" faster? 
            # Actually, distinct sums constraint is about values, not just budget.
            # Let's pick uniformly from the first 10 valid candidates to keep values small-ish 
            # but allow some variation.
            
            if len(candidates) <= 10:
                x = rng.choice(candidates)
            else:
                # Pick from the first 10
                x = rng.choice(candidates[:10])
            
            # Apply move
            tails = [x]
            total = x
            for a in reversed(working_seq):
                total += a
                tails.append(total)
            
            working_seq.append(x)
            working_sums.update(tails)
            
            # Update best if improved
            if len(working_seq) > best_len_local:
                best_len_local = len(working_seq)
                best_found = list(working_seq)
        
        # Update current to the result of this perturbation
        current = working_seq
        
        # If we are stuck at a very short sequence for too long, we might want to restart?
        # The outer loop in construct() handles restarts.
        
        # Check if we improved
        if len(current) < len(best_found):
            # We might have degraded. SA would accept with prob exp(-delta/T).
            # For simplicity, we just keep exploring from this new state.
            # If it's too bad, the next perturbation might fix it.
            pass
    
    return best_found, best_len_local

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    best = []
    best_len = 0
    
    # 1. Get a good initial lower bound using Greedy
    greedy_seq = _greedy_forward([])
    if len(greedy_seq) > best_len:
        best = greedy_seq
        best_len = len(best)
        
    # 2. Run ILS/SA search starting from the greedy result and other seeds
    # We run multiple iterations of ILS with different random seeds to diversify
    
    rng = random.Random(42)
    
    # Run ILS from the greedy solution
    found_seq, found_len = _sa_local_search(greedy_seq, max_val=256, deadline=deadline, best_len=best_len)
    if found_len > best_len:
        best = found_seq
        best_len = found_len
        
    # 3. If time remains, run ILS from other random valid prefixes
    if time.time() < deadline - 5:
        for i in range(10):
            if time.time() > deadline:
                break
            
            # Generate a random valid prefix of length 4-6
            prefix = []
            cur_sums = set()
            last = 0
            try:
                target_len = rng.randint(4, 6)
                for _ in range(target_len):
                    # Pick random next, bias towards smaller
                    found = False
                    # Collect candidates
                    candidates = []
                    for x in range(last + 1, 257):
                        tails = [x]
                        total = x
                        for a in reversed(prefix):
                            total += a
                            tails.append(total)
                        if not cur_sums.intersection(tails):
                            candidates.append(x)
                    
                    if not candidates:
                        break
                    
                    # Pick from first 5 to keep values small
                    if len(candidates) <= 5:
                        x = rng.choice(candidates)
                    else:
                        x = rng.choice(candidates[:5])
                    
                    # Apply
                    tails = [x]
                    total = x
                    for a in reversed(prefix):
                        total += a
                        tails.append(total)
                    
                    prefix.append(x)
                    cur_sums.update(tails)
                    last = x
                    found = True
                
                if len(prefix) >= 3:
                    found_seq, found_len = _sa_local_search(prefix, max_val=256, deadline=deadline, best_len=best_len)
                    if found_len > best_len:
                        best = found_seq
                        best_len = found_len
            except Exception:
                pass

    # Final validation
    if not _is_valid_full(best):
        # Fallback
        best = _greedy_forward([])
        
    return best

# EVOLVE-BLOCK-END
