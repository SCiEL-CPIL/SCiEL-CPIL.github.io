import time
import random

def construct():
    """
    Return a nonempty subset of [1,100] with no subset summing to 200.
    Maximize cardinality.
    
    Strategy:
    1. Bitset-based validity check.
    2. Local search starting from greedy initializations.
    3. Random restarts to escape local optima.
    4. Deadline guard.
    """
    deadline = time.time() + 85.0
    TARGET = 200
    N = 100
    ALL_ELEMENTS = list(range(1, 101))
    
    def compute_reachable(elements):
        """Compute subset sums up to TARGET using bitset operations."""
        reachable = 1
        mask = (1 << (TARGET + 1)) - 1
        for x in elements:
            if x <= TARGET:
                reachable |= (reachable << x)
                reachable &= mask
        return reachable

    def is_valid(elements):
        """Check if no subset sums to TARGET."""
        reachable = compute_reachable(elements)
        return not (reachable & (1 << TARGET))

    def greedy_order(order):
        """Run greedy with given element order, return selected set."""
        selected = []
        reachable = 1
        mask = (1 << (TARGET + 1)) - 1
        for x in order:
            # Check if adding x creates sum TARGET
            # If (TARGET - x) is reachable, then adding x creates sum TARGET
            if x <= TARGET:
                if reachable & (1 << (TARGET - x)):
                    continue
            # Add x
            selected.append(x)
            if x <= TARGET:
                reachable |= (reachable << x)
                reachable &= mask
        return selected

    best_set = []
    best_size = 0
    
    # Phase 1: Deterministic Greedy Initializations
    orders = [
        list(range(1, 101)),
        list(range(100, 0, -1)),
        list(range(50, 0, -1)) + list(range(51, 101)),
        list(range(1, 51)) + list(range(100, 50, -1))
    ]
    
    for order in orders:
        if time.time() > deadline:
            break
        candidate = greedy_order(order)
        if len(candidate) > best_size:
            best_size = len(candidate)
            best_set = candidate[:]
            
    # Phase 2: Local Search with Random Restarts
    # We try to improve the current best set by swapping elements
    # and also try random initializations of size > best_size
    
    random.seed(42)
    
    # Try to find a set of size best_size + 1 or more
    # We will run local search on random sets of size k
    
    max_k_to_try = min(30, N) # Don't try absurdly large sets initially
    
    # First, refine the current best_set
    current_set = set(best_set)
    current_size = best_size
    
    # Local search on the best found so far
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try to add any element
        for x in ALL_ELEMENTS:
            if x in current_set:
                continue
            candidate = list(current_set) + [x]
            if is_valid(candidate):
                current_set.add(x)
                current_size += 1
                best_size = current_size
                best_set = list(current_set)
                improved = True
                break
        
        if not improved:
            # Try swaps: remove one, add one
            # To find a swap that allows further growth or just maintains size
            # We want to increase size, so swaps are neutral.
            # But sometimes a neutral swap allows a subsequent add.
            # We'll do a few random neutral swaps to perturb the state.
            if time.time() > deadline - 5.0:
                break
                
            # Try a limited number of random neutral swaps
            for _ in range(50):
                if time.time() > deadline:
                    break
                if len(current_set) < 2:
                    break
                to_remove = random.choice(list(current_set))
                to_add_candidates = [x for x in ALL_ELEMENTS if x not in current_set]
                if not to_add_candidates:
                    break
                to_add = random.choice(to_add_candidates)
                
                candidate_set = (current_set - {to_remove}) | {to_add}
                if is_valid(list(candidate_set)):
                    # Accept the swap if it doesn't reduce size (it doesn't)
                    # We accept it to explore new neighborhoods
                    current_set = candidate_set
                    # Note: We don't update best_set here unless size increases,
                    # but we keep current_set for further exploration
                    break

    # Phase 3: Random Restarts for larger sets
    # If we haven't found a set of size 20+, try random sets of size 20, 21, etc.
    for target_size in range(best_size + 1, 31):
        if time.time() > deadline - 10.0:
            break
            
        # Try to find a valid set of this size via random sampling and local repair
        found = False
        attempts = 0
        max_attempts = 1000 # Limit attempts per size to avoid timeout
        
        while attempts < max_attempts and time.time() < deadline - 5.0 and not found:
            attempts += 1
            # Random sample of target_size elements
            candidate = random.sample(ALL_ELEMENTS, target_size)
            
            if is_valid(candidate):
                if target_size > best_size:
                    best_size = target_size
                    best_set = candidate
                    found = True
                    break
            
            # Local repair: try to fix invalid set by swapping
            # If it's invalid, find a subset summing to 200 and break it
            # This is complex, so we just rely on random sampling for now
            # or simple local search on the candidate
            
            # Simple local search on the random candidate
            c_set = set(candidate)
            c_size = target_size
            c_improved = True
            local_deadline = time.time() + 0.05 # Very short per candidate
            
            while c_improved and time.time() < local_deadline:
                c_improved = False
                # Try to remove elements that cause conflicts?
                # Actually, if it's invalid, we must remove elements.
                # We want to KEEP size. So this repair logic is tricky.
                # If we remove, size drops.
                # So random sampling is better for finding *valid* sets of fixed size.
                pass

    # Final check: ensure best_set is valid and non-empty
    if not best_set:
        # Fallback: {1} is always valid
        best_set = [1]
        best_size = 1
    elif not is_valid(best_set):
        # Fallback to known valid set {1..19}
        best_set = list(range(1, 20))
        best_size = 19

    return best_set

# EVOLVE-BLOCK-END
