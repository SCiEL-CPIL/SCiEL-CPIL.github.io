import time
import random
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin for 120s limit
    
    N = 512
    MAX_A = N - 1  # 511
    
    # Precompute prime factors for each number 1..511
    prime_factors = [set() for _ in range(N)]
    for a in range(2, N):
        x = a
        p = 2
        while p * p <= x:
            while x % p == 0:
                prime_factors[a].add(p)
                x //= p
            p += 1
        if x > 1:
            prime_factors[a].add(x)
    
    # Precompute compatibility matrix
    # is_coprime[a][b] is True if gcd(a,b)==1
    is_coprime = [[False] * N for _ in range(N)]
    for a in range(1, N):
        for b in range(1, N):
            is_coprime[a][b] = not (prime_factors[a] & prime_factors[b])
    
    # Precompute weight as Fraction for exact arithmetic
    weight = [Fraction(0) for _ in range(N)]
    for a in range(1, N):
        weight[a] = Fraction(1, N - a)
    
    def score(S):
        total = Fraction(0)
        for a in S:
            total += weight[a]
        return total
    
    def greedy_init():
        S = set()
        for a in range(MAX_A, 0, -1):
            if all(is_coprime[a][b] for b in S):
                S.add(a)
        return S
    
    def local_search(S):
        current_score = score(S)
        improved = True
        while improved and (time.time() - start_time) < time_limit - 5.0:
            improved = False
            
            # Try 1-opt swaps
            for a_out in list(S):
                best_a_in = None
                best_gain = Fraction(0)
                S_minus = S - {a_out}
                for a_in in range(MAX_A, 0, -1):
                    if a_in in S:
                        continue
                    if all(is_coprime[a_in][b] for b in S_minus):
                        gain = weight[a_in] - weight[a_out]
                        if gain > best_gain:
                            best_gain = gain
                            best_a_in = a_in
                if best_a_in is not None:
                    S.remove(a_out)
                    S.add(best_a_in)
                    current_score += best_gain
                    improved = True
                    break
            
            # Try 2-opt swaps (remove two, add two)
            if not improved:
                found = False
                S_list = list(S)
                n = len(S_list)
                for i in range(n):
                    for j in range(i+1, n):
                        a_out1 = S_list[i]
                        a_out2 = S_list[j]
                        S_minus2 = S - {a_out1, a_out2}
                        best_pair = None
                        best_gain = Fraction(0)
                        
                        # Try all pairs of additions
                        for a_in1 in range(MAX_A, 0, -1):
                            if a_in1 in S:
                                continue
                            if not all(is_coprime[a_in1][b] for b in S_minus2):
                                continue
                            for a_in2 in range(MAX_A, 0, -1):
                                if a_in2 in S or a_in2 == a_in1:
                                    continue
                                if not all(is_coprime[a_in2][b] for b in S_minus2):
                                    continue
                                if not is_coprime[a_in1][a_in2]:
                                    continue
                                gain = weight[a_in1] + weight[a_in2] - weight[a_out1] - weight[a_out2]
                                if gain > best_gain:
                                    best_gain = gain
                                    best_pair = (a_in1, a_in2)
                        
                        if best_pair is not None:
                            S.remove(a_out1)
                            S.remove(a_out2)
                            S.add(best_pair[0])
                            S.add(best_pair[1])
                            current_score += best_gain
                            improved = True
                            found = True
                            break
                    if found:
                        break
            
            # Try adds
            if not improved:
                best_a_in = None
                best_gain = Fraction(0)
                for a_in in range(MAX_A, 0, -1):
                    if a_in in S:
                        continue
                    if all(is_coprime[a_in][b] for b in S):
                        gain = weight[a_in]
                        if gain > best_gain:
                            best_gain = gain
                            best_a_in = a_in
                if best_a_in is not None:
                    S.add(best_a_in)
                    current_score += best_gain
                    improved = True
        
        return S, current_score
    
    # Run multiple restarts
    best_S = set()
    best_score = Fraction(0)
    
    num_restarts = 0
    while (time.time() - start_time) < time_limit - 15.0:
        num_restarts += 1
        S = greedy_init()
        S, sc = local_search(S)
        if sc > best_score:
            best_score = sc
            best_S = set(S)
    
    # Final polish on best_S
    S = set(best_S)
    S, sc = local_search(S)
    if sc > best_score:
        best_score = sc
        best_S = set(S)
        
    return list(best_S)
