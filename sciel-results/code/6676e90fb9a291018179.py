"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _compute_mults(A):
    """Compute representation multiplicities for all targets in [128, 256].
    A is a set of integers in [0, 128].
    Returns a dict mapping target t -> count of unordered pairs (a,b) in A with a+b=t.
    """
    A_list = sorted(A)
    mults = {}
    for t in range(128, 257):
        count = 0
        for a in A_list:
            b = t - a
            if b in A:
                if a == b:
                    count += 1
                elif a < b:
                    count += 1
        mults[t] = count
    return mults

def _max_mult(A):
    """Return the maximum multiplicity over targets [128, 256]."""
    mults = _compute_mults(A)
    return max(mults.values())

def _is_valid(A):
    """Check that every target in [128, 256] has at least one representation."""
    A_list = sorted(A)
    for t in range(128, 257):
        found = False
        for a in A_list:
            b = t - a
            if b in A:
                if a == b or a < b:
                    found = True
                    break
        if not found:
            return False
    return True

def _greedy_init():
    """Greedy construction: start with empty set, add elements to cover all targets."""
    A = set()
    # We need to cover all sums in [128, 256].
    # A simple approach: start with elements that cover many uncovered targets.
    # But for speed, let's use a structured initial guess and then repair.
    
    # Start with a reasonable base: every other element or similar
    # Actually, let's just start with a random valid set via greedy repair.
    
    # Initialize with a subset that likely covers most targets
    # For sum range [128, 256], if A contains [64, 128], all sums from 128 to 256 are covered.
    # That's 65 elements. Let's try to do better.
    
    # Greedy: pick elements that maximize coverage of currently uncovered targets
    uncovered = set(range(128, 257))
    candidates = list(range(0, 129))
    random.shuffle(candidates)
    
    A = set()
    while uncovered and candidates:
        best_elem = None
        best_gain = 0
        for elem in candidates:
            # How many new targets does adding 'elem' cover?
            gain = 0
            for t in uncovered:
                partner = t - elem
                if 0 <= partner <= 128:
                    if partner in A or partner == elem:
                        gain += 1
            if gain > best_gain:
                best_gain = gain
                best_elem = elem
        if best_elem is not None and best_gain > 0:
            A.add(best_elem)
            candidates.remove(best_elem)
            # Update uncovered
            new_uncovered = set()
            for t in uncovered:
                covered = False
                for a in A:
                    b = t - a
                    if b in A:
                        if a == b or a < b:
                            covered = True
                            break
                if not covered:
                    new_uncovered.add(t)
            uncovered = new_uncovered
        else:
            break
    
    if not uncovered:
        return A
    
    # Fallback: add the full range [64, 128] which is guaranteed to work
    return set(range(64, 129))

def _repair(A):
    """Ensure A is valid by adding elements if needed."""
    A = set(A)
    while not _is_valid(A):
        # Find an uncovered target
        uncovered_targets = []
        for t in range(128, 257):
            found = False
            for a in A:
                b = t - a
                if b in A:
                    if a == b or a < b:
                        found = True
                        break
            if not found:
                uncovered_targets.append(t)
        
        if not uncovered_targets:
            break
        
        # Pick a random uncovered target and try to cover it
        t = random.choice(uncovered_targets)
        # Try adding an element that covers this target
        added = False
        for a in range(0, 129):
            b = t - a
            if 0 <= b <= 128 and (a == b or (a not in A and b not in A)):
                # Adding 'a' might help if 'b' is already in A, or if we add both
                if b in A:
                    A.add(a)
                    added = True
                    break
        if not added:
            # Add a pair
            a = random.randint(0, 128)
            b = t - a
            if 0 <= b <= 128:
                A.add(a)
                A.add(b)
                added = True
        if not added:
            # Fallback: add from [64, 128]
            A.update(range(64, 129))
    
    return A

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85 seconds, leaving 5s margin
    
    # Initialize
    A = _greedy_init()
    A = _repair(A)
    
    if not _is_valid(A):
        # Last resort
        return list(range(64, 129))
    
    best_A = set(A)
    best_score = _max_mult(best_A)
    
    # Simulated annealing
    current_temp = 10.0
    cooling_rate = 0.995
    min_temp = 0.01
    
    while time.time() < deadline and current_temp > min_temp:
        # Generate a neighbor
        move_type = random.random()
        
        if move_type < 0.4 and len(A) > 10:
            # Remove a random element
            elem = random.choice(list(A))
            new_A = set(A)
            new_A.discard(elem)
            if not _is_valid(new_A):
                continue
        elif move_type < 0.8:
            # Swap: remove one, add another
            if len(A) == 0:
                continue
            elem = random.choice(list(A))
            new_elem = random.randint(0, 128)
            new_A = set(A)
            new_A.discard(elem)
            new_A.add(new_elem)
            if not _is_valid(new_A):
                continue
        else:
            # Add a random element
            new_elem = random.randint(0, 128)
            if new_elem in A:
                continue
            new_A = set(A)
            new_A.add(new_elem)
            if not _is_valid(new_A):
                continue
        
        new_score = _max_mult(new_A)
        
        if new_score < best_score:
            best_score = new_score
            best_A = set(new_A)
        
        # Accept/reject
        delta = new_score - _max_mult(A)
        if delta < 0 or random.random() < np.exp(-delta / current_temp):
            A = new_A
        
        current_temp *= cooling_rate
    
    # Final repair and return
    best_A = _repair(best_A)
    if not _is_valid(best_A):
        return list(range(64, 129))
    
    return list(best_A)

# EVOLVE-BLOCK-END
