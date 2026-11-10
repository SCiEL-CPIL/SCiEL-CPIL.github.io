import random
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin for 150s limit
    N = 1000
    MAX_SUM = 2 * N
    
    # Greedy initialization using NumPy for counts
    def greedy_init(order):
        S = set()
        S_list = []
        counts = np.zeros(MAX_SUM + 1, dtype=np.int32)
        
        for x in order:
            valid = True
            if counts[2 * x] >= 2:
                valid = False
            
            if valid and S_list:
                # Check if any count[x + y] >= 2 for y in S_list
                # We can do this efficiently if we had an array of S, but S_list is small initially
                # For larger S, we might need to be careful. 
                # Let's just loop for now in init, it's O(N^2) worst case but N=1000 is small enough for init
                for y in S_list:
                    if counts[x + y] >= 2:
                        valid = False
                        break
            
            if valid:
                S.add(x)
                S_list.append(x)
                # Update counts
                # counts[2*x] += 1
                # counts[x + y] += 1 for y in S_list[:-1]
                # Using numpy slice for the second part? No, x+y is a list of indices.
                # Let's just do it in a loop for clarity in init, it's fast enough.
                for y in S_list[:-1]:
                    counts[x + y] += 1
                counts[2 * x] += 1
                
        return S, S_list, counts

    def local_search(S_init, S_list_init, counts_init, max_iters=100000):
        best_S = set(S_init)
        best_S_list = list(S_list_init)
        best_counts = counts_init.copy()
        best_size = len(S_init)
        
        current_S = set(S_init)
        current_S_list = list(S_list_init)
        current_counts = counts_init.copy()
        
        # Precompute a list of all elements for random selection
        all_elements = np.arange(1, N + 1)
        
        # Helper to try insert
        def try_insert(x):
            if x in current_S:
                return True
            if current_counts[2 * x] >= 2:
                return False
            if current_S_list:
                # Check counts[x + y] for y in current_S_list
                # We can use numpy for this if we have an array of current_S_list
                # But current_S_list changes. Let's just loop.
                for y in current_S_list:
                    if current_counts[x + y] >= 2:
                        return False
            current_S.add(x)
            current_S_list.append(x)
            for y in current_S_list[:-1]:
                current_counts[x + y] += 1
            current_counts[2 * x] += 1
            return True

        def remove_from(x):
            if x not in current_S:
                return False
            current_S.remove(x)
            current_S_list.remove(x)
            for y in current_S_list:
                current_counts[x + y] -= 1
            current_counts[2 * x] -= 1
            return True

        for it in range(max_iters):
            if time.time() > deadline:
                break
                
            # Random move
            if random.random() < 0.3:
                # Try to add a random element
                x = random.randint(1, N)
                if try_insert(x):
                    if len(current_S) > best_size:
                        best_S = set(current_S)
                        best_S_list = list(current_S_list)
                        best_counts = current_counts.copy()
                        best_size = len(current_S)
            else:
                # Swap: remove a random element, try to add a different one
                if not current_S_list:
                    continue
                x_to_remove = random.choice(current_S_list)
                remove_from(x_to_remove)
                
                # Try to add a random element
                x_to_add = random.randint(1, N)
                if x_to_add == x_to_remove:
                    # Put it back
                    try_insert(x_to_remove)
                    continue
                
                if try_insert(x_to_add):
                    if len(current_S) > best_size:
                        best_S = set(current_S)
                        best_S_list = list(current_S_list)
                        best_counts = current_counts.copy()
                        best_size = len(current_S)
                else:
                    # Put the removed element back
                    try_insert(x_to_remove)
                    
        return best_S, best_S_list, best_counts

    random.seed(42)
    best_S_overall = set()
    best_size_overall = 0
    
    # Try a few different orderings for greedy init
    orderings = [
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
        sorted(range(1, N + 1), key=lambda x: random.random()),
    ]
    
    for ordering in orderings:
        if time.time() > deadline:
            break
        S, S_list, counts = greedy_init(ordering)
        best_S, best_S_list, best_counts = local_search(S, S_list, counts, max_iters=50000)
        
        if len(best_S) > best_size_overall:
            best_size_overall = len(best_S)
            best_S_overall = best_S

    # Final polish: try to insert any remaining elements
    if time.time() < deadline - 5:
        final_S = set(best_S_overall)
        final_S_list = list(best_S_overall)
        final_counts = np.zeros(MAX_SUM + 1, dtype=np.int32)
        for x in final_S_list:
            for y in final_S_list:
                if y < x:
                    continue
                final_counts[x + y] += 1
        
        for x in range(1, N + 1):
            if x in final_S:
                continue
            # Check if x can be added
            valid = True
            if final_counts[2 * x] >= 2:
                valid = False
            if valid:
                for y in final_S_list:
                    if final_counts[x + y] >= 2:
                        valid = False
                        break
            if valid:
                final_S.add(x)
                final_S_list.append(x)
                for y in final_S_list[:-1]:
                    final_counts[x + y] += 1
                final_counts[2 * x] += 1
                if len(final_S) > best_size_overall:
                    best_size_overall = len(final_S)
                    best_S_overall = final_S

    return list(best_S_overall)
