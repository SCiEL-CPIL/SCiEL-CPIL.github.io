import math
import time

def construct():
            start_time = time.time()
            deadline = start_time + 85
            
            # Sieve for primes up to 256
            max_sum = 256
            sieve = [True] * (max_sum + 1)
            sieve[0] = sieve[1] = False
            for i in range(2, int(max_sum**0.5) + 1):
                if sieve[i]:
                    for j in range(i*i, max_sum+1, i):
                        sieve[j] = False
            prime_set = set(i for i in range(2, max_sum+1) if sieve[i])
            
            # Precompute valid B for each A, split by parity of B
            valid_b_even = {}  # b is even
            valid_b_odd = {}   # b is odd
            for a in range(1, 129):
                valid_b_even[a] = set()
                valid_b_odd[a] = set()
                for b in range(1, 129):
                    if (a + b) in prime_set:
                        if b % 2 == 0:
                            valid_b_even[a].add(b)
                        else:
                            valid_b_odd[a].add(b)
            
            odd_list = [x for x in range(1, 129) if x % 2 == 1]
            even_list = [x for x in range(1, 129) if x % 2 == 0]
            
            # Case 1: A all odd, B all even (sums are odd)
            for a1 in odd_list:
                if time.time() > deadline:
                    break
                vb1 = valid_b_even[a1]
                if len(vb1) < 3:
                    continue
                for a2 in odd_list:
                    if a2 <= a1:
                        continue
                    if time.time() > deadline:
                        break
                    vb12 = vb1 & valid_b_even[a2]
                    if len(vb12) < 3:
                        continue
                    for a3 in odd_list:
                        if a3 <= a2:
                            continue
                        if time.time() > deadline:
                            break
                        valid_b = vb12 & valid_b_even[a3]
                        if len(valid_b) < 3:
                            continue
                        b_list = sorted(valid_b)
                        for i in range(len(b_list) - 2):
                            for j in range(i + 1, len(b_list) - 1):
                                for k in range(j + 1, len(b_list)):
                                    B = (b_list[i], b_list[j], b_list[k])
                                    sums = [a + b for a in (a1, a2, a3) for b in B]
                                    if len(set(sums)) == 9:
                                        return [[a1, a2, a3], list(B)]
            
            # Case 2: A all even, B all odd (sums are odd)
            for a1 in even_list:
                if time.time() > deadline:
                    break
                vb1 = valid_b_odd[a1]
                if len(vb1) < 3:
                    continue
                for a2 in even_list:
                    if a2 <= a1:
                        continue
                    if time.time() > deadline:
                        break
                    vb12 = vb1 & valid_b_odd[a2]
                    if len(vb12) < 3:
                        continue
                    for a3 in even_list:
                        if a3 <= a2:
                            continue
                        if time.time() > deadline:
                            break
                        valid_b = vb12 & valid_b_odd[a3]
                        if len(valid_b) < 3:
                            continue
                        b_list = sorted(valid_b)
                        for i in range(len(b_list) - 2):
                            for j in range(i + 1, len(b_list) - 1):
                                for k in range(j + 1, len(b_list)):
                                    B = (b_list[i], b_list[j], b_list[k])
                                    sums = [a + b for a in (a1, a2, a3) for b in B]
                                    if len(set(sums)) == 9:
                                        return [[a1, a2, a3], list(B)]
            
            # Fallback: unrestricted search
            valid_b_all = {}
            for a in range(1, 129):
                valid_b_all[a] = set()
                for b in range(1, 129):
                    if (a + b) in prime_set:
                        valid_b_all[a].add(b)
            
            for a1 in range(1, 129):
                if time.time() > deadline:
                    break
                vb1 = valid_b_all[a1]
                for a2 in range(a1 + 1, 129):
                    if time.time() > deadline:
                        break
                    vb12 = vb1 & valid_b_all[a2]
                    if len(vb12) < 3:
                        continue
                    for a3 in range(a2 + 1, 129):
                        if time.time() > deadline:
                            break
                        valid_b = vb12 & valid_b_all[a3]
                        if len(valid_b) < 3:
                            continue
                        b_list = sorted(valid_b)
                        for i in range(len(b_list) - 2):
                            for j in range(i + 1, len(b_list) - 1):
                                for k in range(j + 1, len(b_list)):
                                    B = (b_list[i], b_list[j], b_list[k])
                                    sums = [a + b for a in (a1, a2, a3) for b in B]
                                    if len(set(sums)) == 9:
                                        return [[a1, a2, a3], list(B)]
            
            return [[1, 7, 13], [4, 10, 16]]
