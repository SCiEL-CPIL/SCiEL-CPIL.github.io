import time
import heapq

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    # State: (last_val, depth, suffix_sums, all_sums, seq)
    # Heap ordered by last_val (min-heap) to prioritize smaller last elements
    
    start_state = (0, 0, frozenset([0]), frozenset(), [])
    
    heap = [(0, 0, 0, frozenset([0]), frozenset(), [])]
    # Added a counter '0' at index 2 to avoid comparing frozensets if last_val and depth are equal
    
    best_seq = []
    best_len = 0
    counter = 0
    
    # Pruning cache: key = (suffix_sums, all_sums), value = min_last_val_seen
    seen = {}
    
    while heap:
        if time.time() > deadline:
            break
        
        last_val, depth, _, suffix_sums, all_sums, seq = heapq.heappop(heap)
        
        # Check if this state is still valid (not dominated by a newer, better one)
        key = (suffix_sums, all_sums)
        if key in seen and seen[key] < last_val:
            continue
            
        # Update best
        if depth > best_len:
            best_len = depth
            best_seq = seq[:]
            
        # Prune: mark this state as seen with its last_val
        if key not in seen or last_val < seen[key]:
            seen[key] = last_val
        else:
            # This state was already seen with a smaller or equal last_val, so we skip expansion
            continue

        # Generate next states
        for a in range(last_val + 1, 257):
            # Calculate new suffix sums
            new_suffix_sums = frozenset(a + s for s in suffix_sums)
            
            # Check if new sums collide with existing all_sums
            if not new_suffix_sums.isdisjoint(all_sums):
                continue
                
            # New all_sums = all_sums union new_suffix_sums
            new_all_sums = all_sums | new_suffix_sums
            
            # Check if the new state is dominated
            new_key = (new_suffix_sums, new_all_sums)
            if new_key in seen and seen[new_key] <= a:
                continue
                
            # Push to heap
            counter += 1
            heapq.heappush(heap, (a, depth + 1, counter, new_suffix_sums, new_all_sums, seq + [a]))
            
            # If heap is getting too large, we might run out of time/memory
            # A simple heuristic: if we've expanded many states at this depth and haven't improved best_len, 
            # we might be stuck. But let's rely on the deadline.
            
    return best_seq
