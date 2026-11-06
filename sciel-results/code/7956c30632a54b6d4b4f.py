import numpy as np
import time
import random
import sys

def construct():
    n = 512
    deadline = time.time() + 80.0  # safety margin before 90s kill
    
    # Precompute all 4-APs (a, a+d, a+2d, a+3d) within [1, n]
    # Represent each AP as a tuple of 0-indexed positions
    aps = []
    for a in range(0, n):
        for d in range(1, (n - a) // 3 + 1):
            if a + 3 * d < n:
                aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    n_aps = len(aps)
    
    # For each element, store the indices of APs it participates in
    element_aps = [[] for _ in range(n)]
    for i, ap in enumerate(aps):
        for pos in ap:
            element_aps[pos].append(i)
    
    # Convert to numpy arrays for fast lookup
    # ap_positions: shape (n_aps, 4), int32
    ap_positions = np.array(aps, dtype=np.int32)
    
    # element_aps_arrays: list of int32 arrays
    element_aps_arrays = [np.array(ea, dtype=np.int32) for ea in element_aps]
    
    def greedy_randomized(order=None):
        """
        Greedy construction: process elements in given order (or random),
        add element if it doesn't complete any 4-AP with already-selected elements.
        """
        if order is None:
            order = list(range(n))
            random.shuffle(order)
        
        selected = np.zeros(n, dtype=bool)
        # For each AP, count how many of its 4 positions are selected
        ap_count = np.zeros(n_aps, dtype=np.int32)
        
        for x in order:
            # Check if adding x would complete any AP
            # An AP is completed if it contains x and the other 3 positions are already selected
            # i.e., ap_count[ap_idx] == 3 for some ap_idx in element_aps[x]
            if len(element_aps_arrays[x]) == 0:
                # No APs involve x, always safe
                selected[x] = True
                continue
            
            ap_indices = element_aps_arrays[x]
            # Check if any of these APs already has 3 other positions selected
            # ap_count[ap_indices] gives the count of selected positions in each AP
            # We need to check if any AP has count == 3 (meaning all other 3 are selected)
            # But wait: ap_count counts ALL selected positions in the AP, including x if x is already selected
            # Since we're checking BEFORE adding x, x is not yet selected, so ap_count for APs containing x
            # counts the other selected positions.
            # So if ap_count[ap_idx] == 3, that means all 3 other positions are selected -> adding x completes the AP.
            if np.any(ap_count[ap_indices] == 3):
                continue
            
            # Safe to add x
            selected[x] = True
            # Update ap_count for all APs containing x
            ap_count[ap_indices] += 1
        
        return selected.sum()
    
    def local_search(selected):
        """
        Try to improve by:
        1. Removing elements that block many additions
        2. Adding elements that were previously blocked
        Simple version: try to add any element that can now be added (after removals),
        and try removing elements that are part of many APs with other selected elements.
        """
        selected = selected.copy()
        n_aps_local = len(aps)
        
        # Rebuild ap_count
        ap_count = np.zeros(n_aps_local, dtype=np.int32)
        for x in range(n):
            if selected[x]:
                ap_count[element_aps_arrays[x]] += 1
        
        improved = True
        while improved:
            improved = False
            
            # Phase 1: Try to add any element that can be added
            for x in range(n):
                if selected[x]:
                    continue
                if len(element_aps_arrays[x]) == 0:
                    selected[x] = True
                    ap_count[element_aps_arrays[x]] += 1
                    improved = True
                    continue
                ap_indices = element_aps_arrays[x]
                if not np.any(ap_count[ap_indices] == 3):
                    selected[x] = True
                    ap_count[ap_indices] += 1
                    improved = True
            
            # Phase 2: Try removing elements that are "expensive" (block many potential additions)
            # For each selected element, count how many unselected elements it blocks
            # An element x blocks an unselected element y if there's an AP containing both x and y
            # where the other 2 positions are selected.
            # This is complex; simple heuristic: remove elements that are in many "nearly complete" APs
            # i.e., APs where 3 positions are selected and one is unselected.
            # Actually, let's try a simpler approach: for each selected element, 
            # check if removing it allows adding more elements.
            
            # Try removing each selected element and see if we can add more
            for x in range(n):
                if not selected[x]:
                    continue
                
                # Temporarily remove x
                selected[x] = False
                ap_count[element_aps_arrays[x]] -= 1
                
                # Try to add elements
                added_count = 0
                added_elements = []
                for y in range(n):
                    if selected[y]:
                        continue
                    if len(element_aps_arrays[y]) == 0:
                        selected[y] = True
                        ap_count[element_aps_arrays[y]] += 1
                        added_count += 1
                        added_elements.append(y)
                        continue
                    ap_indices_y = element_aps_arrays[y]
                    if not np.any(ap_count[ap_indices_y] == 3):
                        selected[y] = True
                        ap_count[ap_indices_y] += 1
                        added_count += 1
                        added_elements.append(y)
                
                if added_count > 1:
                    # Net gain: removed 1, added added_count
                    improved = True
                else:
                    # Revert
                    selected[x] = True
                    ap_count[element_aps_arrays[x]] += 1
                    for y in added_elements:
                        selected[y] = False
                        ap_count[element_aps_arrays[y]] -= 1
        
        return selected.sum()
    
    best_size = 0
    best_selected = None
    
    # Phase 1: Many random greedy restarts
    start_time = time.time()
    num_restarts = 0
    
    while time.time() < deadline - 15.0:  # Leave 15s for local search
        order = list(range(n))
        random.shuffle(order)
        
        size = greedy_randomized(order)
        num_restarts += 1
        
        if size > best_size:
            best_size = size
            # Reconstruct the selected set
            order2 = list(range(n))
            random.shuffle(order2)
            # Actually, we need to keep the same order. Let's redo this properly.
            pass
    
    # The above has a bug: we don't save the selected array. Let me fix.
    
    best_size = 0
    best_selected = None
    num_restarts = 0
    
    while time.time() < deadline - 20.0:
        order = list(range(n))
        random.shuffle(order)
        
        selected = np.zeros(n, dtype=bool)
        ap_count = np.zeros(n_aps, dtype=np.int32)
        
        for x in order:
            if len(element_aps_arrays[x]) == 0:
                selected[x] = True
                continue
            
            ap_indices = element_aps_arrays[x]
            if np.any(ap_count[ap_indices] == 3):
                continue
            
            selected[x] = True
            ap_count[ap_indices] += 1
        
        size = selected.sum()
        num_restarts += 1
        
        if size > best_size:
            best_size = size
            best_selected = selected.copy()
    
    # Phase 2: Local search on the best solution
    if best_selected is not None and time.time() < deadline - 5.0:
        improved_size = local_search(best_selected)
        if improved_size > best_size:
            best_size = improved_size
    
    # Also try local search on a few more random solutions if time permits
    if time.time() < deadline - 5.0:
        for _ in range(5):
            if time.time() > deadline - 5.0:
                break
            order = list(range(n))
            random.shuffle(order)
            
            selected = np.zeros(n, dtype=bool)
            ap_count = np.zeros(n_aps, dtype=np.int32)
            
            for x in order:
                if len(element_aps_arrays[x]) == 0:
                    selected[x] = True
                    continue
                ap_indices = element_aps_arrays[x]
                if np.any(ap_count[ap_indices] == 3):
                    continue
                selected[x] = True
                ap_count[ap_indices] += 1
            
            size = selected.sum()
            if size > best_size - 2:  # Only try local search if close to best
                ls_size = local_search(selected)
                if ls_size > best_size:
                    best_size = ls_size
    
    # Return the best solution
    if best_selected is not None:
        result = [i + 1 for i in range(n) if best_selected[i]]
    else:
        # Fallback: simple greedy
        result = [1]
    
    return sorted(result)

# Set a fixed seed for reproducibility in testing, but the evaluator will call construct()
# which uses random module. We don't set a global seed to allow diversity.
