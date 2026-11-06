import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    elements = list(range(1, 23))
    L = 232792560
    vals = {s: L // s for s in elements}
    
    best_set = []
    best_size = 0
    
    # Greedy initialization
    greedy_set = []
    greedy_sums = {0}
    for s in elements:
        v = vals[s]
        if v not in greedy_sums:
            new_sums = set()
            for sum_val in greedy_sums:
                new_sums.add(sum_val + v)
                new_sums.add(sum_val - v)
            if len(new_sums) == 2 * len(greedy_sums):
                greedy_set.append(s)
                greedy_sums = new_sums
                
    best_set = greedy_set
    best_size = len(greedy_set)
    
    # DFS
    current_set = []
    current_sums = {0}
    
    def dfs(index):
        nonlocal best_set, best_size
        
        if time.time() > deadline:
            return
            
        if len(current_set) + (len(elements) - index) <= best_size:
            return
            
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = list(current_set)
            
        if index == len(elements):
            return
            
        # Exclude
        dfs(index + 1)
        
        # Include
        s = elements[index]
        v = vals[s]
        
        if v not in current_sums:
            new_sums = set()
            for sum_val in current_sums:
                new_sums.add(sum_val + v)
                new_sums.add(sum_val - v)
                
            if len(new_sums) == 2 * len(current_sums):
                current_set.append(s)
                old_sums = current_sums
                current_sums = new_sums
                
                dfs(index + 1)
                
                current_set.pop()
                current_sums = old_sums
                
    dfs(0)
    
    return best_set
