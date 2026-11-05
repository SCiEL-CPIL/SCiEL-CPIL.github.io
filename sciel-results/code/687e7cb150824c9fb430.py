import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    min_sum = 128
    max_sum = 256
    n = 128
    
    # Initial Set: The seed [64..128]
    best_A = set(range(64, 129))
    
    def get_counts(A):
        A_list = sorted(A)
        counts = [0] * (max_sum + 1)
        for i in range(len(A_list)):
            a = A_list[i]
            for j in range(i, len(A_list)):
                b = A_list[j]
                s = a + b
                if min_sum <= s <= max_sum:
                    counts[s] += 1
        return counts

    def is_valid(counts):
        for s in range(min_sum, max_sum + 1):
            if counts[s] == 0:
                return False
        return True

    def get_score(counts):
        if not is_valid(counts):
            return float('inf')
        return max(counts[min_sum:max_sum+1])

    best_counts = get_counts(best_A)
    best_score = get_score(best_counts)
    
    current_A = set(best_A)
    current_counts = best_counts[:]
    current_score = best_score
    
    T = 1.0
    T_min = 0.01
    cooling = 0.999 # Slower cooling for more exploration
    
    # Precompute list of A for faster iteration? 
    # A changes, so we just use list(current_A)
    
    while time.time() < deadline:
        if len(current_A) == 0:
            break
            
        move_type = random.random()
        
        if move_type < 0.8:
            # Swap
            x = random.choice(list(current_A))
            y = random.randint(0, n)
            while y in current_A:
                y = random.randint(0, n)
                
            new_counts = current_counts[:]
            
            # Remove x
            for z in current_A:
                s = x + z
                if min_sum <= s <= max_sum:
                    new_counts[s] -= 1
                    
            # Add y
            for z in current_A:
                if z == x:
                    continue
                s = y + z
                if min_sum <= s <= max_sum:
                    new_counts[s] += 1
                    
            if not is_valid(new_counts):
                continue
                
            new_score = max(new_counts[min_sum:max_sum+1])
            delta = new_score - current_score
            
            if delta < 0 or random.random() < pow(2.71828, -delta / T):
                current_A.remove(x)
                current_A.add(y)
                current_counts = new_counts
                current_score = new_score
                
                if current_score < best_score:
                    best_score = current_score
                    best_A = set(current_A)
                    
        elif move_type < 0.9:
            # Add
            y = random.randint(0, n)
            if y in current_A:
                continue
                
            new_counts = current_counts[:]
            for z in current_A:
                s = y + z
                if min_sum <= s <= max_sum:
                    new_counts[s] += 1
                    
            if not is_valid(new_counts):
                continue
                
            new_score = max(new_counts[min_sum:max_sum+1])
            delta = new_score - current_score
            
            if delta < 0 or random.random() < pow(2.71828, -delta / T):
                current_A.add(y)
                current_counts = new_counts
                current_score = new_score
                
                if current_score < best_score:
                    best_score = current_score
                    best_A = set(current_A)
                    
        else:
            # Remove
            x = random.choice(list(current_A))
            new_counts = current_counts[:]
            for z in current_A:
                s = x + z
                if min_sum <= s <= max_sum:
                    new_counts[s] -= 1
                    
            if not is_valid(new_counts):
                continue
                
            new_score = max(new_counts[min_sum:max_sum+1])
            delta = new_score - current_score
            
            if delta < 0 or random.random() < pow(2.71828, -delta / T):
                current_A.remove(x)
                current_counts = new_counts
                current_score = new_score
                
                if current_score < best_score:
                    best_score = current_score
                    best_A = set(current_A)
                    
        T *= cooling
        if T < T_min:
            T = T_min

    return list(best_A)
