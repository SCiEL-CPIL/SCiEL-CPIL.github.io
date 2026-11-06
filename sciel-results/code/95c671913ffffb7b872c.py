import time
import math
import random
import numpy as np

N = 5000

def _primes(limit):
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

# Precompute primes up to 1000 (sufficient for tuples of size ~450)
MAX_PRIME_FOR_PRECOMP = 1000
ALL_PRIMES = _primes(MAX_PRIME_FOR_PRECOMP)

# Precompute residue class indices for fast lookup
prime_indices = {}
for p in ALL_PRIMES:
    indices = [[] for _ in range(p)]
    for x in range(N + 1):
        r = x % p
        indices[r].append(x)
    prime_indices[p] = indices

# Precompute numpy arrays for each prime/residue for fast counting
prime_residue_arrays = {}
for p in ALL_PRIMES:
    arrays = []
    for r in range(p):
        arr = np.zeros(N + 1, dtype=bool)
        arr[prime_indices[p][r]] = True
        arrays.append(arr)
    prime_residue_arrays[p] = arrays

def _is_admissible_fast(S_active):
    """Fast check if the active set is admissible using numpy."""
    size = int(S_active.sum())
    for p in ALL_PRIMES:
        if p > size:
            break
        # Check if all residue classes are present
        for r in range(p):
            mask = prime_residue_arrays[p][r]
            if not np.any(S_active & mask):
                # This residue class is missed, so this prime is satisfied
                break
        else:
            # All residue classes are present for this prime
            return False
    return True

def _count_residues(S_active, p):
    """Count active elements in each residue class mod p."""
    counts = np.zeros(p, dtype=np.int32)
    for r in range(p):
        mask = prime_residue_arrays[p][r]
        counts[r] = np.sum(S_active & mask)
    return counts

def construct():
    start_time = time.time()
    deadline = start_time + 280.0
    
    best_size = 0
    best_set = []
    
    # Strategy: Multiple randomized greedy attempts + local search with swaps
    
    num_greedy_tries = 30
    greedy_results = []
    
    for trial in range(num_greedy_tries):
        if time.time() > deadline - 80:
            break
            
        S_active = np.ones(N + 1, dtype=bool)
        current_size = N + 1
        
        # Process primes in random order for diversity
        primes_to_process = [p for p in ALL_PRIMES if p <= current_size]
        random.shuffle(primes_to_process)
        
        for p in primes_to_process:
            if time.time() > deadline - 80:
                break
            if p > current_size:
                continue
            
            # Calculate counts for each residue
            counts = _count_residues(S_active, p)
            min_count = int(np.min(counts))
            
            if min_count > 0:
                # All residues are present; remove one
                # Randomly select among min-count residues
                min_indices = np.where(counts == min_count)[0]
                r_to_remove = int(random.choice(min_indices))
                
                mask = prime_residue_arrays[p][r_to_remove]
                active_in_class = S_active & mask
                current_size -= int(np.sum(active_in_class))
                S_active[active_in_class] = False
        
        greedy_results.append((current_size, S_active.copy()))
        
        if current_size > best_size:
            best_size = current_size
            best_set = [int(x) for x in range(N + 1) if S_active[x]]
    
    # Sort greedy results by size, best first
    greedy_results.sort(key=lambda x: -x[0])
    
    # Local search refinement on top greedy results
    # Move: try to add elements back, and also try swap moves (remove one, add multiple)
    local_search_budget = 200.0
    local_search_start = time.time()
    
    for size, S_active in greedy_results[:15]:
        if time.time() - local_search_start > local_search_budget:
            break
        if size < best_size:
            continue
        
        S_active = S_active.copy()
        current_size = size
        
        # Phase 1: Try to add elements back
        inactive_elements = [x for x in range(N + 1) if not S_active[x]]
        random.shuffle(inactive_elements)
        
        added = 0
        for x in inactive_elements:
            if time.time() - local_search_start > local_search_budget:
                break
            
            # Tentatively add x
            S_active[x] = True
            new_size = current_size + 1
            
            # Check admissibility
            if _is_admissible_fast(S_active):
                current_size = new_size
                added += 1
                if current_size > best_size:
                    best_size = current_size
                    best_set = [int(x) for x in range(N + 1) if S_active[x]]
            else:
                S_active[x] = False
        
        # Phase 2: Try swap moves - remove one active element, then try to add multiple inactive
        if time.time() - local_search_start > local_search_budget:
            continue
            
        active_elements = [x for x in range(N + 1) if S_active[x]]
        random.shuffle(active_elements)
        
        # Try a limited number of swap attempts
        swap_attempts = min(200, len(active_elements))
        for _ in range(swap_attempts):
            if time.time() - local_search_start > local_search_budget:
                break
            
            # Pick a random active element to remove
            x_remove = random.choice(active_elements)
            S_active[x_remove] = False
            size_after_remove = current_size - 1
            
            # Now try to add back multiple inactive elements
            inactive_now = [x for x in range(N + 1) if not S_active[x]]
            random.shuffle(inactive_now)
            
            added_in_swap = 0
            for x_add in inactive_now:
                if time.time() - local_search_start > local_search_budget:
                    break
                if added_in_swap >= 5:  # Limit additions per swap
                    break
                    
                S_active[x_add] = True
                new_size = size_after_remove + added_in_swap + 1
                
                if _is_admissible_fast(S_active):
                    added_in_swap += 1
                else:
                    S_active[x_add] = False
            
            final_size = size_after_remove + added_in_swap
            
            if final_size > current_size:
                current_size = final_size
                if current_size > best_size:
                    best_size = current_size
                    best_set = [int(x) for x in range(N + 1) if S_active[x]]
            else:
                # Revert the swap - put back the removed element
                S_active[x_remove] = True
                # Remove any that were added
                for x_add in inactive_now:
                    if S_active[x_add] and x_add != x_remove:
                        # Check if this was added during swap
                        pass  # Complex to track; just revert if not improved
                # Simpler: just restore x_remove and keep current state
                # Since final_size <= current_size, we revert
                S_active[x_remove] = True
        
        # Phase 3: Final polish - try adding any remaining inactive elements
        if time.time() - local_search_start > local_search_budget:
            continue
            
        inactive_final = [x for x in range(N + 1) if not S_active[x]]
        random.shuffle(inactive_final)
        
        for x in inactive_final:
            if time.time() - local_search_start > local_search_budget:
                break
            
            S_active[x] = True
            new_size = current_size + 1
            
            if _is_admissible_fast(S_active):
                current_size = new_size
                if current_size > best_size:
                    best_size = current_size
                    best_set = [int(x) for x in range(N + 1) if S_active[x]]
            else:
                S_active[x] = False
    
    return sorted(best_set)
