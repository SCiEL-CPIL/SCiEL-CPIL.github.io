import time
import random
import math
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    N = 256
    
    # Helper to compute all block sums for a given sequence
    def get_block_sums(seq):
        sums = set()
        n = len(seq)
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += seq[j]
                sums.add(s)
        return sums

    # Helper to check validity and return the set of sums if valid
    def check_and_get_sums(seq):
        if not seq:
            return True, set()
        sums = set()
        n = len(seq)
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += seq[j]
                if s in sums:
                    return False, None
                sums.add(s)
        return True, sums

    # Initial seed: Greedy descending
    def greedy_seed():
        seq = []
        used = set()
        # Start with large numbers to leave room for small numbers later?
        # Or start with small? 
        # Descending greedy usually works well for this type of constraint.
        for x in range(N, 0, -1):
            new_seq = seq + [x]
            new_seq.sort()
            valid, _ = check_and_get_sums(new_seq)
            if valid:
                seq = new_seq
                used.add(x)
        return seq

    best_seq = greedy_seed()
    best_len = len(best_seq)
    
    # If greedy already found a very long sequence, we might need a different start
    # But let's start SA from the greedy result or a random valid subset.
    
    # Strategy: Simulated Annealing on a set of indices.
    # State: A set of selected indices from [1, 256].
    # Objective: Maximize len(state) subject to validity.
    # Moves:
    # 1. Add a random unselected index.
    # 2. Remove a random selected index.
    # 3. Swap a selected index with an unselected index.
    
    # To make SA efficient, we maintain the set of block sums.
    # Adding an element x at position p in sorted seq:
    # New sums are:
    # - x
    # - x + suffixes of seq[p:]
    # - prefixes of seq[:p] + x
    # - prefixes of seq[:p] + x + suffixes of seq[p:]
    # Actually, it's easier to just recompute if the sequence is small (< 30).
    # For N=256, max length is likely around 15-20.
    # O(k^2) is ~ 400 ops. Very fast.
    
    # Let's use a simpler approach: 
    # State is a sorted list.
    # We keep a set of current block sums.
    # When adding x, we check for collisions.
    # If collision, we reject (or in SA, accept with prob exp(-delta/T)).
    # Delta = number of collisions? Or just binary valid/invalid?
    # If we allow invalid states, we need a penalty.
    # Better: Only consider valid states. 
    # But then we can't "jump" over barriers if adding x causes collision.
    # To escape local optima, we need to remove elements.
    # So: Move is "Add x" (if valid) OR "Remove x" (always valid) OR "Swap x, y" (if valid).
    # If "Add x" is invalid, we don't take it. This is like Random Walk on Valid States.
    # To maximize length, we bias towards Add.
    # If we get stuck (no adds possible), we force a Remove to open space.
    
    # Let's implement a "Biased Random Walk with Forced Removals"
    # which is effectively SA with T=0 for validity but random restarts/local search.
    # Actually, let's do proper SA with a penalty for invalidity to allow "tunneling".
    
    # Energy function:
    # E = -len(seq) + C * (number of collisions)
    # We want to minimize E.
    
    def evaluate(seq):
        """Returns (length, collision_count)"""
        if not seq:
            return 0, 0
        sums = set()
        collisions = 0
        n = len(seq)
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += seq[j]
                if s in sums:
                    collisions += 1
                else:
                    sums.add(s)
        return len(seq), collisions

    def energy(seq):
        l, c = evaluate(seq)
        # Strong penalty for collisions to keep search mostly in valid region
        # But allow some invalid states to jump barriers
        return -l + 10 * c

    # Initial state
    current_seq = best_seq[:]
    current_energy = energy(current_seq)
    
    # We will run multiple restarts
    # Or just one long SA with cooling.
    
    # Let's try a few restarts with different initial states
    # 1. Greedy Descending
    # 2. Greedy Ascending
    # 3. Random Valid Subset
    
    def random_valid_subset(min_size=5):
        seq = []
        attempts = 0
        while len(seq) < min_size and attempts < 100:
            x = random.randint(1, N)
            if x not in seq:
                new_seq = seq + [x]
                new_seq.sort()
                valid, _ = check_and_get_sums(new_seq)
                if valid:
                    seq = new_seq
            attempts += 1
        return seq

    starts = [
        best_seq,
        greedy_seed(), # Just in case, though it's the same
        random_valid_subset(10),
        random_valid_subset(10)
    ]
    
    # Filter valid starts
    valid_starts = []
    for s in starts:
        v, _ = check_and_get_sums(s)
        if v:
            valid_starts.append(s)
    
    if not valid_starts:
        return best_seq

    # For each start, run SA
    for start_seq in valid_starts:
        if time.time() > deadline - 5:
            break
            
        current = start_seq[:]
        current_len = len(current)
        current_energy = energy(current)
        
        best_local = current[:]
        best_local_len = current_len
        best_local_energy = current_energy
        
        T = 10.0  # Initial temperature
        T_min = 0.1
        cooling_rate = 0.9995
        
        # Steps per restart
        # We have ~80s. Each step is fast.
        # Let's run until time limit or T < T_min
        
        while time.time() < deadline - 2:
            if T < T_min:
                break
                
            # Choose move
            # 1. Add (50%)
            # 2. Remove (20%)
            # 3. Swap (30%)
            
            move_type = random.random()
            new_seq = current[:]
            
            if move_type < 0.5:
                # Add
                available = [x for x in range(1, N+1) if x not in set(new_seq)]
                if available:
                    x = random.choice(available)
                    # Insert in sorted position
                    pos = 0
                    while pos < len(new_seq) and new_seq[pos] < x:
                        pos += 1
                    new_seq.insert(pos, x)
            elif move_type < 0.7:
                # Remove
                if new_seq:
                    idx = random.randint(0, len(new_seq)-1)
                    new_seq.pop(idx)
            else:
                # Swap
                if new_seq:
                    # Pick one to remove
                    idx_remove = random.randint(0, len(new_seq)-1)
                    x_remove = new_seq[idx_remove]
                    
                    # Pick one to add
                    available = [x for x in range(1, N+1) if x not in set(new_seq)]
                    if available:
                        x_add = random.choice(available)
                        new_seq[idx_remove] = x_add
                        new_seq.sort()
            
            new_energy = energy(new_seq)
            
            # Acceptance
            delta_E = new_energy - current_energy
            
            if delta_E < 0:
                # Better (lower energy)
                current = new_seq
                current_energy = new_energy
                
                if current_energy < best_local_energy or (current_energy == best_local_energy and len(current) > best_local_len):
                    # Actually, we track best by length primarily, but energy handles validity
                    # If valid, energy = -len. So lower energy means longer (if valid).
                    # If invalid, energy = -len + 10*c.
                    # We want the best VALID sequence.
                    # So we should only update best_local if the new state is VALID.
                    v, _ = check_and_get_sums(current)
                    if v:
                        best_local = current[:]
                        best_local_energy = current_energy
                        best_local_len = len(current)
                        
            else:
                # Worse
                # Accept with probability exp(-delta_E / T)
                if delta_E == 0:
                    prob = 1.0
                else:
                    prob = math.exp(-delta_E / T)
                
                if random.random() < prob:
                    current = new_seq
                    current_energy = new_energy
                    # Do not update best_local if invalid, but we can track best valid seen so far
                    # Actually, if we accept a worse state, it might be invalid.
                    # We only update best_local if current is valid.
                    v, _ = check_and_get_sums(current)
                    if v:
                        if len(current) > best_local_len:
                            best_local = current[:]
                            best_local_len = len(current)
                            best_local_energy = current_energy
                            
            T *= cooling_rate
            
        # End of SA for this start
        if best_local_len > best_len:
            best_len = best_local_len
            best_seq = best_local[:]
            
    # Final check
    v, _ = check_and_get_sums(best_seq)
    if not v:
        # Fallback to greedy
        best_seq = greedy_seed()
        
    return best_seq

if __name__ == "__main__":
    res = construct()
    print(len(res))
