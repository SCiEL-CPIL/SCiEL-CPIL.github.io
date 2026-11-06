from fractions import Fraction

def construct():
    def find_solution(k, max_nodes=50000):
        nodes_visited = [0]
        
        def dfs(idx, current_sum, last_denom):
            if nodes_visited[0] > max_nodes:
                return None
            
            if idx == k:
                if current_sum == 1:
                    return []
                return None
            
            if current_sum > 1:
                return None
            
            R = 1 - current_sum
            if R <= 0:
                return None
            
            # d_min
            # d >= 1/R
            F = 1 / R
            if F.denominator == 1:
                d_min = max(last_denom + 1, F.numerator)
            else:
                d_min = max(last_denom + 1, F.numerator // F.denominator + 1)
            
            # d_max
            # d <= (k - idx + 1) / R
            # Also d > last_denom
            G = (k - idx + 1) / R
            if G.denominator == 1:
                d_max = G.numerator
            else:
                d_max = G.numerator // G.denominator
            
            if d_max < d_min:
                return None
            
            # Limit d_max to prevent huge ranges if R is tiny
            # But for small k, R is not tiny initially.
            # Cap d_max to 10000 for safety in early stages?
            # If d is very large, 1/d is small.
            # For k=7, d should be small.
            
            for d in range(d_min, d_max + 1):
                if 1 / d > R:
                    continue
                
                new_sum = current_sum + Fraction(1, d)
                if new_sum > 1:
                    continue
                
                nodes_visited[0] += 1
                if nodes_visited[0] > max_nodes:
                    return None
                
                rem_terms = k - idx - 1
                rem_sum = 1 - new_sum
                
                if rem_terms == 0:
                    if rem_sum == 0:
                        return [d]
                    else:
                        continue
                
                # Prune: Max sum of rem_terms starting from d+1
                # sum_{i=1}^{rem_terms} 1/(d+i) < rem_terms / (d+1)
                if rem_terms / (d + 1) < rem_sum:
                    break
                
                sub_sol = dfs(idx + 1, new_sum, d)
                if sub_sol is not None:
                    return [d] + sub_sol
            
            return None
        
        return dfs(0, Fraction(0, 1), 4)

    for k in range(6, 12):
        sol = find_solution(k)
        if sol:
            return sol
    
    # Fallback
    return [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256]

