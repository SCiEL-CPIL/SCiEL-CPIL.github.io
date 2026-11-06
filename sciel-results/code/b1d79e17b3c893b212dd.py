import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    N = 512
    MAX_SIZE = 48
    MAX_SUM = N * MAX_SIZE
    
    def count_violations(S):
        """Count mean violations in set S."""
        n = len(S)
        if n < 3:
            return 0
        
        violations = 0
        for i, x in enumerate(S):
            others = [S[j] for j in range(n) if j != i]
            m = len(others)
            if m < 2:
                continue
            
            dp = [0] * (m + 1)
            dp[0] = 1
            for e in others:
                for k in range(m, 0, -1):
                    dp[k] |= (dp[k-1] << e)
            
            for k in range(2, m + 1):
                target = k * x
                if target > MAX_SUM:
                    break
                if (dp[k] >> target) & 1:
                    violations += 1
                    break
        
        return violations
    
    # Simulated annealing
    random.seed(123)
    
    # Start with a medium-sized random set
    initial_size = 30
    current = set(random.sample(range(1, N + 1), initial_size))
    current_cost = count_violations(list(current))
    
    best = set(current)
    best_cost = current_cost
    best_size = len(current)
    
    T = 50.0
    T_min = 0.5
    cooling = 0.998
    
    sa_deadline = min(deadline, start_time + 75.0)
    
    while time.time() < sa_deadline and T > T_min:
        # Generate neighbor
        if len(current) > 0:
            action = random.random()
            if action < 0.7:
                # Swap: remove one, add one
                to_remove = random.choice(list(current))
                to_add = random.randint(1, N)
                if to_add not in current:
                    neighbor = set(current)
                    neighbor.remove(to_remove)
                    neighbor.add(to_add)
                else:
                    neighbor = set(current)
            elif action < 0.85:
                # Add one
                to_add = random.randint(1, N)
                if to_add not in current and len(current) < MAX_SIZE:
                    neighbor = set(current)
                    neighbor.add(to_add)
                else:
                    neighbor = set(current)
            else:
                # Remove one
                if len(current) > 10:
                    to_remove = random.choice(list(current))
                    neighbor = set(current)
                    neighbor.remove(to_remove)
                else:
                    neighbor = set(current)
            
            neighbor_cost = count_violations(list(neighbor))
            delta = neighbor_cost - current_cost
            
            if delta <= 0 or random.random() < math.exp(-delta / T):
                current = neighbor
                current_cost = neighbor_cost
                
                if current_cost == 0 and len(current) > best_size:
                    best = set(current)
                    best_size = len(current)
                    best_cost = 0
        else:
            break
        
        T *= cooling
    
    # Refine: try to grow the best valid set
    if best_cost == 0 and best_size < MAX_SIZE:
        S = list(best)
        grow_deadline = min(deadline, start_time + 83.0)
        
        while time.time() < grow_deadline and len(S) < MAX_SIZE:
            improved = False
            for _ in range(100):
                if time.time() > grow_deadline:
                    break
                x = random.randint(1, N)
                if x not in S:
                    test_S = S + [x]
                    if count_violations(test_S) == 0:
                        S.append(x)
                        improved = True
                        break
            if not improved:
                break
        
        if len(S) > best_size:
            best_size = len(S)
            best = set(S)
    
    return list(best)

if __name__ == "__main__":
    result = construct()
    print(f"Set size: {len(result)}")
    print(result[:10], "...", result[-10:])
