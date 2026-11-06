import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s limit, 5s buffer
    
    n = 256
    # Initial greedy construction
    selected = []
    selected_set = set()
    
    # Greedy: add x if it doesn't form AP with any pair in selected
    for x in range(1, n + 1):
        # Check if x forms AP with any a, b in selected
        valid = True
        for a in selected:
            # Check if 2*a - x is in selected (and != x)
            # Or 2*x - a is in selected (and != x)
            # Or a + x = 2*b for some b?
            # If we add x, we form AP if:
            # 1. There exists a, b in selected such that a+b=2x (x is midpoint)
            # 2. There exists a, b in selected such that a+x=2b (b is midpoint) -> 2b-a=x
            # 3. There exists a, b in selected such that x+b=2a (a is midpoint) -> 2a-b=x
            
            # Since we check against all a in selected:
            # If 2*a - x is in selected_set and 2*a - x != x:
            #   Then x, a, 2a-x form AP? No, 2a-x, x, a?
            #   Midpoint of (2a-x, a) is (3a-x)/2.
            #   Let's just check all 3 permutations or simply:
            #   Does {x, a, b} form an AP?
            #   Sort x, a, b. Check if diff is constant.
            
            # Efficient check:
            # For a given a, potential partners b are:
            # b = 2*a - x (if b in selected and b != x) -> AP: b, a, x? No.
            # If b, a, x are in AP, then 2a = b+x => b = 2a-x.
            # If a, b, x are in AP, then 2b = a+x => b = (a+x)/2.
            # If a, x, b are in AP, then 2x = a+b => b = 2x-a.
            
            # So for each a in selected:
            # 1. Check if b = 2*a - x is in selected_set and b != x.
            #    If so, {b, a, x} is AP.
            # 2. Check if (a+x) is even. If so, b = (a+x)//2.
            #    Check if b is in selected_set and b != x and b != a.
            #    If so, {a, b, x} is AP.
            # 3. Check if b = 2*x - a is in selected_set and b != x.
            #    If so, {a, x, b} is AP.
            
            b1 = 2 * a - x
            if b1 in selected_set and b1 != x:
                valid = False
                break
            
            if (a + x) % 2 == 0:
                b2 = (a + x) // 2
                if b2 in selected_set and b2 != x and b2 != a:
                    valid = False
                    break
            
            b3 = 2 * x - a
            if b3 in selected_set and b3 != x:
                valid = False
                break
        
        if valid:
            selected.append(x)
            selected_set.add(x)
        
        if time.time() > deadline:
            break

    # Local Search: Swap (single and multi-removal)
    # We try to replace one or more elements in selected with v not in selected
    # such that the net reciprocal sum increases and the new set is valid.
    
    def _find_blocking_pairs(v, sel_set):
        """Find all pairs (a,b) in sel_set that form a 3-AP with v."""
        pairs = []
        for a in sel_set:
            # Case 1: b, a, v in AP => 2a = b+v => b = 2a-v
            b1 = 2 * a - v
            if b1 in sel_set and b1 != v:
                pairs.append((b1, a))
            # Case 2: a, b, v in AP => 2b = a+v => b = (a+v)/2
            if (a + v) % 2 == 0:
                b2 = (a + v) // 2
                if b2 in sel_set and b2 != v and b2 != a:
                    pairs.append((a, b2))
            # Case 3: a, v, b in AP => 2v = a+b => b = 2v-a
            b3 = 2 * v - a
            if b3 in sel_set and b3 != v:
                pairs.append((a, b3))
        return pairs

    def _is_valid_after_removals(v, sel_set, to_remove):
        """Check if v can be added after removing elements in to_remove."""
        remaining = sel_set - to_remove
        return len(_find_blocking_pairs(v, remaining)) == 0

    def _try_add_direct(v, sel_set):
        """Try to add v directly if no conflicts."""
        if len(_find_blocking_pairs(v, sel_set)) == 0:
            return True
        return False

    def _try_single_swap(v, sel_set):
        """Try to add v by removing exactly one element. Returns the element to remove or None."""
        blocking = _find_blocking_pairs(v, sel_set)
        if not blocking:
            return None  # Can add directly
        
        # Find element that appears in ALL blocking pairs
        first_pair = blocking[0]
        for u in first_pair:
            if all(u in pair for pair in blocking):
                return u
        return None

    def _try_multi_swap(v, sel_set, max_removals=2):
        """Try to add v by removing up to max_removals elements. Returns set to remove or None."""
        blocking = _find_blocking_pairs(v, sel_set)
        if not blocking:
            return None  # Can add directly
        
        # For 2-removal: try all pairs of elements from the first blocking pair
        # and check if removing them resolves all conflicts
        first_pair = blocking[0]
        candidates = set()
        for a in first_pair:
            for b in first_pair:
                if a != b:
                    candidates.add(frozenset([a, b]))
        
        # Also try pairs where one element is from first pair and another from other pairs
        all_elements = set()
        for pair in blocking:
            all_elements.update(pair)
        
        # Try all pairs from all_elements (limit for efficiency)
        elem_list = list(all_elements)
        if len(elem_list) <= 20:
            for i in range(len(elem_list)):
                for j in range(i+1, len(elem_list)):
                    candidates.add(frozenset([elem_list[i], elem_list[j]]))
        
        # Sort candidates by sum of reciprocals removed (we want to remove large numbers)
        # Actually, we want to minimize loss: remove elements with largest 1/x (smallest x)
        # But we also need v < removed elements for net gain
        # Let's just try all and pick the best valid one
        
        best_removal = None
        best_gain = -1
        
        for removal_set in candidates:
            if len(removal_set) > max_removals:
                continue
            
            # Check if removing this set resolves all conflicts
            if _is_valid_after_removals(v, sel_set, removal_set):
                # Calculate net gain
                gain = 1.0 / v - sum(1.0 / u for u in removal_set)
                if gain > best_gain:
                    best_gain = gain
                    best_removal = removal_set
        
        if best_removal is not None and best_gain > 0:
            return set(best_removal)
        
        return None

    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Sort candidates by value (smaller is better)
        candidates = [v for v in range(1, n + 1) if v not in selected_set]
        
        for v in candidates:
            if time.time() > deadline:
                break
            
            # First try direct addition
            if _try_add_direct(v, selected_set):
                selected.append(v)
                selected_set.add(v)
                improved = True
                break  # Restart search with new set
            
            # Try single swap
            u_to_remove = _try_single_swap(v, selected_set)
            if u_to_remove is not None and v < u_to_remove:
                selected_set.remove(u_to_remove)
                selected.remove(u_to_remove)
                selected.append(v)
                selected_set.add(v)
                improved = True
                break  # Restart search with new set
            
            # Try multi-swap (2 removals)
            removal_set = _try_multi_swap(v, selected_set, max_removals=2)
            if removal_set is not None:
                for u in removal_set:
                    selected_set.remove(u)
                    selected.remove(u)
                selected.append(v)
                selected_set.add(v)
                improved = True
                break  # Restart search with new set
            
            if improved:
                break

    return selected

# EVOLVE-BLOCK-END
