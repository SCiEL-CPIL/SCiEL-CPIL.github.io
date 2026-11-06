from collections import deque

def construct():
    def apply_T(b):
        return [11 if x == 11 else (x + 1) % 11 for x in b]
        
    def apply_S(b):
        return [11 if x == 11 else (2 * x) % 11 for x in b]
        
    def apply_I(b):
        inv_map = {1:1, 2:6, 3:4, 4:3, 5:9, 6:2, 7:8, 8:7, 9:5, 10:10}
        res = []
        for x in b:
            if x == 11:
                res.append(0)
            elif x == 0:
                res.append(11)
            else:
                res.append(inv_map[x])
        return res

    generators = [apply_T, apply_S, apply_I]
    
    seed = [0, 1, 2, 4, 5, 10]
    seed.sort()
    
    visited = set()
    queue = deque([tuple(seed)])
    visited.add(tuple(seed))
    
    while queue:
        current = queue.popleft()
        for gen in generators:
            new_block = gen(list(current))
            new_block.sort()
            new_tuple = tuple(new_block)
            if new_tuple not in visited:
                visited.add(new_tuple)
                queue.append(new_tuple)
                
    blocks = [list(b) for b in visited]
    
    # If we didn't get 132, try to fill with greedy? 
    # But the task says 132 is attained ONLY by the Witt design.
    # If we got < 132, we are suboptimal.
    # However, if the seed is correct and PGL(2,11) is transitive, we get 132.
    
    return blocks
