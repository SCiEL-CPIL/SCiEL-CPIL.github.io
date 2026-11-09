N = 12

def construct():
        # Letter 0: Cyclic shift i -> (i + 1) % N
        sigma0 = [(i + 1) % N for i in range(N)]
        
        # Letter 1: 0 -> 0, i -> i - 1 for i > 0
        sigma1 = [0] * N
        for i in range(1, N):
            sigma1[i] = i - 1
            
        return [sigma0, sigma1]
