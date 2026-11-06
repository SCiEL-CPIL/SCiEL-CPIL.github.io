import numpy as np
import time

def construct():
    N = 51975
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # Generate divisors of 51975 = 3^3 * 5^2 * 7 * 11
    divisors = []
    for a in range(4):
        for b in range(3):
            for c in range(2):
                for d in range(2):
                    d_val = (3 ** a) * (5 ** b) * (7 ** c) * (11 ** d)
                    if d_val > 1:
                        divisors.append(d_val)
    divisors.sort()
    divisors_set = set(divisors)

    # Precompute residue masks for each (m, r) pair
    # mask[m][r] is a boolean array of length N
    masks = {}
    for m in divisors:
        masks[m] = []
        for r in range(m):
            mask = np.zeros(N, dtype=bool)
            mask[r::m] = True
            masks[m].append(mask)

    def count_uncovered(uncovered):
        return np.count_nonzero(uncovered)

    def find_best_residue(m, uncovered):
        """Find the residue r for modulus m that covers the most uncovered residues."""
        max_count = 0
        best_r = 0
        for r in range(m):
            c = np.count_nonzero(uncovered & masks[m][r])
            if c > max_count:
                max_count = c
                best_r = r
        return best_r, max_count

    def greedy_initialization(order=None):
        """Greedy set cover initialization with optional custom order."""
        uncovered = np.ones(N, dtype=bool)
        classes = []
        available_moduli = order if order else divisors.copy()
        
        while available_moduli:
            if time.time() > deadline:
                break
                
            best_gain = 0
            best_m = None
            best_r = None
            
            for m in available_moduli:
                max_count = 0
                best_r_m = 0
                for r in range(m):
                    c = np.count_nonzero(uncovered & masks[m][r])
                    if c > max_count:
                        max_count = c
                        best_r_m = r
                
                if max_count > best_gain:
                    best_gain = max_count
                    best_m = m
                    best_r = best_r_m
            
            if best_m is None or best_gain == 0:
                break
                
            classes.append([best_m, best_r])
            uncovered &= ~masks[best_m][best_r]
            available_moduli.remove(best_m)
            
            if not np.any(uncovered):
                break

        return classes, uncovered

    def coordinate_descent(classes, uncovered, max_iter=20):
        """Coordinate descent: for each selected m, try to find a better r."""
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter and time.time() < deadline:
            improved = False
            iter_count += 1
            current_uncovered_count = count_uncovered(uncovered)
            
            for i in range(len(classes)):
                if time.time() > deadline:
                    break
                m = classes[i][0]
                r_old = classes[i][1]
                
                # Temporarily remove the current residue class
                uncovered |= masks[m][r_old]
                
                # Find best new r for this modulus
                r_new, gain_new = find_best_residue(m, uncovered)
                
                # Apply new r
                new_uncovered = uncovered & ~masks[m][r_new]
                new_uncovered_count = count_uncovered(new_uncovered)
                
                # Check if improvement
                if new_uncovered_count < current_uncovered_count:
                    uncovered = new_uncovered
                    classes[i][1] = r_new
                    current_uncovered_count = new_uncovered_count
                    improved = True
                else:
                    # Revert to original
                    uncovered &= ~masks[m][r_old]
                
                if current_uncovered_count == 0:
                    return classes, uncovered

        return classes, uncovered

    def swap_search(classes, uncovered, unselected_moduli, max_swaps=30):
        """Try to swap a selected modulus with an unselected one."""
        selected_moduli = {c[0] for c in classes}
        improved = False
        current_uncovered_count = count_uncovered(uncovered)
        
        for _ in range(max_swaps):
            if time.time() > deadline:
                break
            
            best_swap_gain = 0
            best_m_out = None
            best_m_in = None
            best_r_in = None
            best_idx = None
            
            for i, (m_out, r_out) in enumerate(classes):
                if time.time() > deadline:
                    break
                
                # Temporarily remove m_out
                new_uncovered = uncovered | masks[m_out][r_out]
                new_uncovered_count = count_uncovered(new_uncovered)
                lost = new_uncovered_count - current_uncovered_count
                
                # Try each unselected modulus
                for m_in in unselected_moduli:
                    r_in, gain_in = find_best_residue(m_in, new_uncovered)
                    net_gain = gain_in - lost
                    if net_gain > best_swap_gain:
                        best_swap_gain = net_gain
                        best_m_out = m_out
                        best_m_in = m_in
                        best_r_in = r_in
                        best_idx = i
                
                if time.time() > deadline:
                    break
            
            if best_m_out is not None and best_swap_gain > 0:
                # Perform swap
                m_out = classes[best_idx][0]
                r_out = classes[best_idx][1]
                classes[best_idx][0] = best_m_in
                classes[best_idx][1] = best_r_in
                
                # Update uncovered
                uncovered = (uncovered | masks[m_out][r_out]) & ~masks[best_m_in][best_r_in]
                
                # Update moduli sets
                selected_moduli.remove(m_out)
                selected_moduli.add(best_m_in)
                unselected_moduli.remove(best_m_in)
                unselected_moduli.append(m_out)
                
                current_uncovered_count = count_uncovered(uncovered)
                improved = True
                if current_uncovered_count == 0:
                    return classes, uncovered, True
            else:
                break

        return classes, uncovered, improved

    def add_unselected(classes, uncovered, unselected_moduli):
        """Add unselected moduli if they help."""
        improved = False
        current_uncovered_count = count_uncovered(uncovered)
        
        for m in unselected_moduli[:]:
            if time.time() > deadline:
                break
            r, gain = find_best_residue(m, uncovered)
            if gain > 0:
                classes.append([m, r])
                uncovered &= ~masks[m][r]
                unselected_moduli.remove(m)
                improved = True
                current_uncovered_count = count_uncovered(uncovered)
                if current_uncovered_count == 0:
                    return classes, uncovered, True

        return classes, uncovered, improved

    def remove_useless(classes, uncovered):
        """Remove classes that don't cover any unique uncovered residues."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(classes)):
                if time.time() > deadline:
                    break
                m = classes[i][0]
                r = classes[i][1]
                # Check if removing this class increases uncovered count
                temp_uncovered = uncovered | masks[m][r]
                if count_uncovered(temp_uncovered) == count_uncovered(uncovered):
                    classes.pop(i)
                    improved = True
                    break

        return classes, uncovered

    # Main algorithm with multiple strategies
    best_classes = None
    best_uncovered_count = N  # Start with worst case
    
    # Strategy 1: Standard greedy
    classes, uncovered = greedy_initialization()
    uncovered_count = count_uncovered(uncovered)
    if uncovered_count < best_uncovered_count:
        best_uncovered_count = uncovered_count
        best_classes = classes.copy()
    
    if best_uncovered_count == 0:
        return best_classes

    # Coordinate descent
    classes, uncovered = coordinate_descent(classes, uncovered, max_iter=15)
    uncovered_count = count_uncovered(uncovered)
    if uncovered_count < best_uncovered_count:
        best_uncovered_count = uncovered_count
        best_classes = classes.copy()
    
    if best_uncovered_count == 0:
        return best_classes

    # Prepare for swap search
    selected_moduli = {c[0] for c in best_classes}
    unselected_moduli = [m for m in divisors if m not in selected_moduli]
    
    # Swap search
    classes, uncovered, _ = swap_search(best_classes, uncovered, unselected_moduli, max_swaps=20)
    uncovered_count = count_uncovered(uncovered)
    if uncovered_count < best_uncovered_count:
        best_uncovered_count = uncovered_count
        best_classes = classes.copy()
    
    if best_uncovered_count == 0:
        return best_classes

    # Add unselected moduli
    classes, uncovered, _ = add_unselected(best_classes, uncovered, unselected_moduli)
    uncovered_count = count_uncovered(uncovered)
    if uncovered_count < best_uncovered_count:
        best_uncovered_count = uncovered_count
        best_classes = classes.copy()
    
    if best_uncovered_count == 0:
        return best_classes

    # Remove useless classes
    classes, uncovered = remove_useless(best_classes, uncovered)
    uncovered_count = count_uncovered(uncovered)
    if uncovered_count < best_uncovered_count:
        best_uncovered_count = uncovered_count
        best_classes = classes.copy()
    
    if best_uncovered_count == 0:
        return best_classes

    # Final coordinate descent
    classes, uncovered = coordinate_descent(best_classes, uncovered, max_iter=10)
    uncovered_count = count_uncovered(uncovered)
    if uncovered_count < best_uncovered_count:
        best_uncovered_count = uncovered_count
        best_classes = classes.copy()
    
    if best_uncovered_count == 0:
        return best_classes

    # Strategy 2: Reverse order greedy
    if time.time() < deadline - 30:
        classes2, uncovered2 = greedy_initialization(order=divisors[::-1])
        uncovered_count2 = count_uncovered(uncovered2)
        if uncovered_count2 < best_uncovered_count:
            best_uncovered_count = uncovered_count2
            best_classes = classes2.copy()
        
        if best_uncovered_count == 0:
            return best_classes

        # Coordinate descent
        classes2, uncovered2 = coordinate_descent(classes2, uncovered2, max_iter=10)
        uncovered_count2 = count_uncovered(uncovered2)
        if uncovered_count2 < best_uncovered_count:
            best_uncovered_count = uncovered_count2
            best_classes = classes2.copy()
        
        if best_uncovered_count == 0:
            return best_classes

    # Strategy 3: Randomized greedy
    if time.time() < deadline - 20:
        np.random.seed(42)
        classes3, uncovered3 = greedy_initialization(order=divisors[:])
        # Shuffle the order
        np.random.shuffle(classes3)
        # Re-run coordinate descent on the shuffled solution
        classes3, uncovered3 = coordinate_descent(classes3, uncovered3, max_iter=15)
        uncovered_count3 = count_uncovered(uncovered3)
        if uncovered_count3 < best_uncovered_count:
            best_uncovered_count = uncovered_count3
            best_classes = classes3.copy()
        
        if best_uncovered_count == 0:
            return best_classes

    # Final cleanup
    best_classes, _ = remove_useless(best_classes, uncovered)

    return best_classes
