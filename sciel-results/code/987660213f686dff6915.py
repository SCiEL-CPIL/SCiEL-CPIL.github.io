import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    N = 512
    start_time = time.time()
    time_limit = 85.0  # Leave 5s margin for evaluator overhead

    # --- Step 1: Precompute all 4-APs as a 3-uniform hypergraph ---
    # A 4-AP is (a, a+d, a+2d, a+3d). If we have a, a+d, a+2d, we cannot have a+3d.
    # So the "forbidden triples" are (a, a+d, a+2d) -> forbids a+3d.
    # We can represent this as: for each element x, which triples of other elements, if all present, forbid x.
    # Actually, it's easier to think: if we add x, we create a 4-AP if there exist a, d such that
    # x is one of the 4 terms, and the other 3 are already in the set.
    
    # Let's use a different representation for speed:
    # For each element x (0-indexed), precompute a list of "conflict triples".
    # A conflict triple for x is a set of 3 elements {u, v, w} such that if u, v, w are in the set,
    # x cannot be added (because {x, u, v, w} forms a 4-AP).
    
    # How to generate:
    # Case 1: x is the 4th term (a+3d). Then the triple is (a, a+d, a+2d).
    #   x = a + 3d => a = x - 3d. Need a >= 0.
    # Case 2: x is the 3rd term (a+2d). Then the triple is (a, a+d, a+3d).
    #   x = a + 2d => a = x - 2d. Need a >= 0 and a+3d < N.
    # Case 3: x is the 2nd term (a+d). Then the triple is (a, a+2d, a+3d).
    #   x = a + d => a = x - d. Need a >= 0 and a+3d < N.
    # Case 4: x is the 1st term (a). Then the triple is (a+d, a+2d, a+3d).
    #   x = a. Need a+3d < N.

    conflict_triples = [[] for _ in range(N)]
    
    for a in range(N):
        for d in range(1, (N - a) // 3 + 1):
            p0, p1, p2, p3 = a, a + d, a + 2 * d, a + 3 * d
            if p3 >= N:
                break
            # If p0, p1, p2 are in set, p3 is forbidden
            conflict_triples[p3].append((p0, p1, p2))
            # If p0, p1, p3 are in set, p2 is forbidden
            conflict_triples[p2].append((p0, p1, p3))
            # If p0, p2, p3 are in set, p1 is forbidden
            conflict_triples[p1].append((p0, p2, p3))
            # If p1, p2, p3 are in set, p0 is forbidden
            conflict_triples[p0].append((p1, p2, p3))

    # Convert to numpy arrays for vectorized checking
    # For each x, store the triples as a 2D array of shape (num_triples_x, 3)
    triple_arrays = []
    for x in range(N):
        if conflict_triples[x]:
            arr = np.array(conflict_triples[x], dtype=np.int32)
            triple_arrays.append(arr)
        else:
            triple_arrays.append(np.empty((0, 3), dtype=np.int32))

    def can_add_batch(mask_bits, candidates):
        """
        mask_bits: numpy array of bools (length N), indicating which elements are in the set.
        candidates: list of element indices to test.
        Returns: list of booleans, same length as candidates, indicating if each can be added.
        """
        results = []
        for x in candidates:
            arr = triple_arrays[x]
            if len(arr) == 0:
                results.append(True)
                continue
            # Check if any row in arr has all 3 elements set in mask_bits
            # arr is shape (k, 3). mask_bits[arr] is shape (k, 3).
            # We need any row where all 3 are True.
            sub = mask_bits[arr]
            if np.any(np.all(sub, axis=1)):
                results.append(False)
            else:
                results.append(True)
        return results

    def count_bits(mask_bits):
        return int(np.sum(mask_bits))

    # --- Step 2: Initial Construction ---
    # Strategy: Use a structured base set.
    # A known construction for 4-AP-free sets is based on digits in base 3 or similar.
    # However, for N=512, a simple greedy on a "good" order might be better.
    # Let's try a "modular" heuristic: keep numbers whose base-3 representation has no '2's?
    # No, that's for 3-APs (Cantor set). For 4-APs, it's less straightforward.
    # Let's stick to greedy but with a smart order: process elements in an order that
    # maximizes "freedom". Elements with fewer conflicts first?
    
    # Calculate conflict degree for each element
    degrees = [len(triple_arrays[i]) for i in range(N)]
    
    # Order 1: Ascending degree (least constrained first)
    order1 = sorted(range(N), key=lambda x: degrees[x])
    # Order 2: Descending degree (most constrained first? maybe to avoid them later)
    order2 = sorted(range(N), key=lambda x: -degrees[x])
    # Order 3: Natural
    order3 = list(range(N))
    # Order 4: Reverse Natural
    order4 = list(range(N-1, -1, -1))
    
    best_mask = None
    best_size = -1
    
    for order in [order1, order2, order3, order4]:
        mask_bits = np.zeros(N, dtype=bool)
        for x in order:
            if can_add_batch(mask_bits, [x])[0]:
                mask_bits[x] = True
        sz = count_bits(mask_bits)
        if sz > best_size:
            best_size = sz
            best_mask = mask_bits.copy()
            
    # Also try random orders
    random.seed(42)
    for _ in range(10):
        if time.time() - start_time > time_limit:
            break
        order = list(range(N))
        random.shuffle(order)
        mask_bits = np.zeros(N, dtype=bool)
        for x in order:
            if can_add_batch(mask_bits, [x])[0]:
                mask_bits[x] = True
        sz = count_bits(mask_bits)
        if sz > best_size:
            best_size = sz
            best_mask = mask_bits.copy()

    current_mask = best_mask
    current_size = best_size

    # --- Step 3: Vectorized Local Search ---
    # Phase A: Try to add any missing element.
    # Instead of checking one by one, check all missing elements at once.
    
    def get_missing(mask_bits):
        return np.where(~mask_bits)[0]
    
    def get_present(mask_bits):
        return np.where(mask_bits)[0]
    
    # Phase A: Additions
    while time.time() - start_time < time_limit:
        missing = get_missing(current_mask)
        if len(missing) == 0:
            break
        
        # Check all missing elements in one batch
        can_add_flags = can_add_batch(current_mask, missing)
        
        # Add all that can be added
        added = False
        for i, x in enumerate(missing):
            if can_add_flags[i]:
                current_mask[x] = True
                current_size += 1
                added = True
        
        if not added:
            break
    
    # Phase B: Swaps (1-out, 1-in or 1-out, k-in)
    # This is harder to vectorize perfectly, but we can do:
    # For each present element y, remove it, then check all missing elements.
    # If any missing element can be added, accept the swap if it increases size (i.e., we added >= 1).
    # Since we remove 1, we need to add >= 1 to not decrease, and >= 2 to increase.
    
    present = get_present(current_mask)
    
    # Shuffle present to randomize
    random.shuffle(present)
    
    for y in present:
        if time.time() - start_time > time_limit:
            break
        
        # Temporarily remove y
        current_mask[y] = False
        
        missing = get_missing(current_mask)
        if len(missing) == 0:
            current_mask[y] = True
            continue
            
        can_add_flags = can_add_batch(current_mask, missing)
        
        # Count how many can be added
        num_addable = int(np.sum(can_add_flags))
        
        if num_addable >= 1:
            # Accept swap: add all addable elements
            for i, x in enumerate(missing):
                if can_add_flags[i]:
                    current_mask[x] = True
                    current_size += 1
            # We removed 1, added num_addable
            current_size -= 1  # Account for the removal of y
            # Wait, the logic above:
            # We set current_mask[y] = False.
            # Then we added num_addable elements.
            # So net change is num_addable - 1.
            # My code above:
            # current_size starts at old_size.
            # I set mask[y]=False. (size effectively old_size-1, but variable still old_size)
            # I loop and add. For each add, current_size += 1.
            # So after loop, current_size = old_size + num_addable.
            # But we removed y, so it should be old_size - 1 + num_addable.
            # So I need to subtract 1.
            current_size -= 1
            
            if current_size > best_size:
                best_size = current_size
                best_mask = current_mask.copy()
                
            # Update present list? It's expensive. Let's just break and restart the loop?
            # Or continue? The set has changed.
            # Let's just break and restart the main while loop to re-evaluate additions.
            break
        else:
            # Revert
            current_mask[y] = True

    # Phase C: 2-out, k-in swaps (more aggressive)
    # Try removing 2 random elements, then adding all possible.
    # This is O(N^2) in the worst case if done naively, but we can sample.
    
    while time.time() - start_time < time_limit:
        present = get_present(current_mask)
        if len(present) < 2:
            break
            
        # Try a few random pairs
        improved = False
        for _ in range(50):
            if time.time() - start_time > time_limit:
                break
                
            y1, y2 = random.sample(present, 2)
            
            # Remove y1, y2
            current_mask[y1] = False
            current_mask[y2] = False
            
            missing = get_missing(current_mask)
            if len(missing) == 0:
                current_mask[y1] = True
                current_mask[y2] = True
                continue
                
            can_add_flags = can_add_batch(current_mask, missing)
            num_addable = int(np.sum(can_add_flags))
            
            if num_addable >= 2:
                # Accept
                for i, x in enumerate(missing):
                    if can_add_flags[i]:
                        current_mask[x] = True
                        current_size += 1
                current_size -= 2 # Removed 2
                
                if current_size > best_size:
                    best_size = current_size
                    best_mask = current_mask.copy()
                improved = True
                break # Restart loop
            else:
                # Revert
                current_mask[y1] = True
                current_mask[y2] = True
        
        if not improved:
            # No improvement in this round of 2-swaps.
            # Try 1-swap again? Or break?
            # Let's try 1-swap one more time.
            present = get_present(current_mask)
            random.shuffle(present)
            found_improvement = False
            for y in present:
                if time.time() - start_time > time_limit:
                    break
                current_mask[y] = False
                missing = get_missing(current_mask)
                if len(missing) == 0:
                    current_mask[y] = True
                    continue
                can_add_flags = can_add_batch(current_mask, missing)
                num_addable = int(np.sum(can_add_flags))
                if num_addable >= 1:
                    for i, x in enumerate(missing):
                        if can_add_flags[i]:
                            current_mask[x] = True
                            current_size += 1
                    current_size -= 1
                    if current_size > best_size:
                        best_size = current_size
                        best_mask = current_mask.copy()
                    found_improvement = True
                    break
                else:
                    current_mask[y] = True
            
            if not found_improvement:
                # Stuck. Try a full restart?
                # Or just break.
                break

    # Final check: Ensure no 4-AP exists.
    # The construction logic ensures this, but let's verify on the best_mask.
    # If invalid, fall back to a simpler valid set (e.g., first 100 numbers? No, must be 4-AP free).
    # A safe fallback: numbers with base 3 digits only 0 or 1.
    # Max such number < 512: 111111_3 = 1+3+9+27+81+243 = 364. Next is 1111111_3 = 1093 > 512.
    # So all numbers in [1, 364] with base 3 digits in {0,1} are valid.
    # Count: 2^6 = 64 numbers. This is a valid, nonempty set.
    
    # Verify best_mask
    def is_valid(mask_bits):
        for a in range(N):
            for d in range(1, (N - a) // 3 + 1):
                p0, p1, p2, p3 = a, a + d, a + 2 * d, a + 3 * d
                if p3 >= N:
                    break
                if mask_bits[p0] and mask_bits[p1] and mask_bits[p2] and mask_bits[p3]:
                    return False
        return True

    if not is_valid(best_mask):
        # Fallback to Cantor-like set
        fallback_mask = np.zeros(N, dtype=bool)
        for i in range(N):
            # Check if base 3 representation of i+1 has only 0s and 1s
            val = i + 1
            valid = True
            while val > 0:
                rem = val % 3
                if rem == 2:
                    valid = False
                    break
                val //= 3
            if valid:
                fallback_mask[i] = True
        if not np.any(fallback_mask):
            # Should not happen, but just in case
            fallback_mask[0] = True
        best_mask = fallback_mask

    # Convert to list
    result = [int(i) + 1 for i in np.where(best_mask)[0]]
    return sorted(result)

# EVOLVE-BLOCK-END
