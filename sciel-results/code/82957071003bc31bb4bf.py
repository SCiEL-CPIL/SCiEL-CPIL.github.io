"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def construct():
    """
    Find a large Sidon set in [1, 284] containing {1, 2, 5, 11}.
    Uses CSP formulation with bitsets for sums and randomized backtracking.
    """
    required = [1, 2, 5, 11]
    N = 284
    
    # Bitset for sums. Max sum is 284+284=568.
    # We use a Python integer as a bitset.
    
    def get_initial_sums(elements):
        sums = 0
        n = len(elements)
        for i in range(n):
            for j in range(i, n):
                s = elements[i] + elements[j]
                sums |= (1 << s)
        return sums

    def can_add(elements, used_sums, x):
        """
        Check if x can be added to elements given used_sums bitset.
        Returns True if valid.
        """
        # Check 2*x
        if (used_sums >> (2 * x)) & 1:
            return False
        
        # Check x + e for e in elements
        for e in elements:
            s = x + e
            if (used_sums >> s) & 1:
                return False
        
        return True

    def add_element(elements, used_sums, x):
        """
        Add x to elements and update used_sums.
        Returns new_elements, new_sums.
        """
        new_elements = elements + [x]
        new_sums = used_sums | (1 << (2 * x))
        for e in elements:
            new_sums |= (1 << (x + e))
        return new_elements, new_sums

    best_size = len(required)
    best_set = list(required)
    best_sums = get_initial_sums(required)
    
    # Time limit management
    start_time = time.time()
    time_limit = 115.0  # Leave 5s buffer
    
    def time_exceeded():
        return time.time() - start_time > time_limit

    def backtrack(elements, used_sums, candidates):
        """
        Randomized backtracking with pruning.
        candidates: list of available candidates.
        """
        nonlocal best_size, best_set, best_sums
        
        if len(elements) > best_size:
            best_size = len(elements)
            best_set = list(elements)
            best_sums = used_sums
            
        if not candidates:
            return
        
        if time_exceeded():
            return

        # Pruning: if even adding all remaining candidates doesn't beat best, stop
        if len(elements) + len(candidates) <= best_size:
            return

        # Find all valid candidates
        valid_candidates = []
        for x in candidates:
            if can_add(elements, used_sums, x):
                valid_candidates.append(x)
                
        if not valid_candidates:
            return
            
        # Pruning again with valid candidates
        if len(elements) + len(valid_candidates) <= best_size:
            return

        # Heuristic: Shuffle to explore different paths
        random.shuffle(valid_candidates)
        
        for x in valid_candidates:
            if time_exceeded():
                return
                
            new_elements, new_sums = add_element(elements, used_sums, x)
            # Remove x from candidates
            new_candidates = [c for c in candidates if c != x]
            
            backtrack(new_elements, new_sums, new_candidates)
            
            # If we found a perfect solution (max possible), we can stop
            if best_size >= 20:
                return

    # Run multiple restarts with different random seeds/orders
    base_elements = sorted(required)
    base_sums = get_initial_sums(base_elements)
    all_candidates = [x for x in range(1, N + 1) if x not in required]
    
    for i in range(200): # Max 200 restarts, but time limited
        if time_exceeded():
            break
            
        # Shuffle candidates for this restart
        current_candidates = list(all_candidates)
        random.shuffle(current_candidates)
        
        backtrack(base_elements, base_sums, current_candidates)
        
        if best_size >= 20:
            break
            
    # Final greedy improvement
    current = sorted(best_set)
    current_sums = get_initial_sums(current)
    remaining = [x for x in range(1, N + 1) if x not in current]
    random.shuffle(remaining)
    for x in remaining:
        if can_add(current, current_sums, x):
            current, current_sums = add_element(current, current_sums, x)
            
    if len(current) > best_size:
        best_set = current
        best_size = len(current)

    # Final validation
    result = sorted(best_set)
    sums = set()
    valid = True
    for i in range(len(result)):
        for j in range(i, len(result)):
            s = result[i] + result[j]
            if s in sums:
                valid = False
                break
            sums.add(s)
        if not valid:
            break
            
    if not valid or not all(x in result for x in required):
        result = list(required)
        
    return result

# EVOLVE-BLOCK-END
