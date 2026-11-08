import time
import random
import math

def construct():
        start_time = time.time()
        deadline = start_time + 100 # 100s limit
        
        def score(U):
            if not U: return 0
            max_u = max(U)
            if max_u == 0:
                return 1.0
            denom = math.log(2 * max_u + 1)
            if denom == 0: return 1.0
            
            u_plus = set()
            u_minus = set()
            for a in U:
                for b in U:
                    u_plus.add(a + b)
                    u_minus.add(a - b)
            
            size_plus = len(u_plus)
            size_minus = len(u_minus)
            
            if size_plus == 0: return 1.0
            ratio = size_minus / size_plus
            if ratio <= 1:
                return 1.0
            return 1 + math.log(ratio) / denom

        best_set = [0, 1, 3, 9]
        best_score = score(best_set)
        
        # Generate candidates
        candidates = []
        # Base 3
        for k in range(3, 8):
            candidates.append([0] + [3**i for i in range(1, k)])
        # Base 2
        for k in range(3, 8):
            candidates.append([0] + [2**i for i in range(1, k)])
        # Base 4
        for k in range(3, 6):
            candidates.append([0] + [4**i for i in range(1, k)])
            
        # Random
        for _ in range(100):
            n = random.randint(3, 8)
            max_val = random.randint(10, 100)
            s = random.sample(range(1, max_val), n-1)
            s.sort()
            candidates.append([0] + s)
            
        for c in candidates:
            s = score(c)
            if s > best_score:
                best_score = s
                best_set = c
        
        # Local Search
        current_set = best_set[:]
        current_score = best_score
        
        while time.time() < deadline:
            improved = False
            # Try modifying each element
            for i in range(len(current_set)):
                if i == 0: continue # Keep 0 fixed? Or allow? Task says containing 0.
                # Try random new value
                for _ in range(10):
                    old_val = current_set[i]
                    # Random new value
                    new_val = random.randint(1, 200)
                    if new_val == old_val: continue
                    
                    # Ensure uniqueness and sorted order?
                    # Set must be a set of integers. Duplicates don't change the set.
                    # We should maintain it as a sorted list of unique integers.
                    new_set_list = current_set[:]
                    new_set_list[i] = new_val
                    # Remove duplicates and sort
                    new_set_list = sorted(list(set(new_set_list)))
                    
                    # Check if 0 is still there
                    if 0 not in new_set_list:
                        continue
                    
                    s = score(new_set_list)
                    if s > current_score:
                        current_score = s
                        current_set = new_set_list
                        improved = True
                        break # Restart loop? Or just continue?
            
            if not improved:
                # Try adding or removing an element?
                # Or random restart?
                # Simple: try a random perturbation of the whole set
                n = len(current_set)
                new_set_list = current_set[:]
                # Perturb one element significantly
                idx = random.randint(1, n-1)
                new_set_list[idx] = random.randint(1, 300)
                new_set_list = sorted(list(set(new_set_list)))
                if 0 in new_set_list:
                    s = score(new_set_list)
                    if s > current_score:
                        current_score = s
                        current_set = new_set_list
                        improved = True
            
            if not improved:
                # Add a new element?
                if len(current_set) < 10:
                    new_val = random.randint(1, 300)
                    if new_val not in current_set:
                        new_set_list = current_set[:] + [new_val]
                        new_set_list = sorted(new_set_list)
                        s = score(new_set_list)
                        if s > current_score:
                            current_score = s
                            current_set = new_set_list
                            improved = True
                
                if not improved:
                    # Remove an element?
                    if len(current_set) > 3:
                        idx = random.randint(1, len(current_set)-1)
                        new_set_list = current_set[:idx] + current_set[idx+1:]
                        s = score(new_set_list)
                        if s > current_score:
                            current_score = s
                            current_set = new_set_list
                            improved = True
            
            if not improved:
                # Random restart with a different seed structure
                base = random.choice([2, 3, 4, 5])
                k = random.randint(3, 6)
                new_set_list = [0] + [base**i for i in range(1, k)]
                # Perturb
                for i in range(1, len(new_set_list)):
                    if random.random() < 0.5:
                        new_set_list[i] += random.randint(-2, 2)
                new_set_list = sorted(list(set([x for x in new_set_list if x >= 0])))
                if 0 in new_set_list:
                    s = score(new_set_list)
                    if s > current_score:
                        current_score = s
                        current_set = new_set_list

        return sorted(current_set)
