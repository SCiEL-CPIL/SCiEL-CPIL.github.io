import time
from fractions import Fraction

def construct():
    target = Fraction(5, 31)
    
    low = 7
    high = 200
    
    def find_solution(target, M):
        suffix_sums = [0.0] * (M + 2)
        for i in range(M, 6, -1):
            suffix_sums[i] = suffix_sums[i+1] + 1.0/i
            
        solution = []
        
        def dfs(rem, min_d):
            if rem == 0:
                return True
            if rem < 0:
                return False
            if min_d > M:
                return False
            if float(rem) > suffix_sums[min_d] + 1e-12:
                return False
                
            # Iterate from M down to min_d
            for d in range(M, min_d - 1, -1):
                new_rem = rem - Fraction(1, d)
                
                if new_rem < 0:
                    # 1/d is too large. Since d decreases, 1/d increases.
                    # So all subsequent (smaller d) will also have 1/d too large.
                    break
                
                if new_rem == 0:
                    solution.append(d)
                    return True
                
                if new_rem > 0:
                    solution.append(d)
                    if dfs(new_rem, d + 1):
                        return True
                    solution.pop()
                    
            return False
            
        if dfs(target, 7):
            return solution
        return None

    while low < high:
        mid = (low + high) // 2
        if find_solution(target, mid):
            high = mid
        else:
            low = mid + 1
            
    sol = find_solution(target, low)
    if sol:
        return sol
    else:
        return [8, 31, 248]

if __name__ == "__main__":
    print(construct())
