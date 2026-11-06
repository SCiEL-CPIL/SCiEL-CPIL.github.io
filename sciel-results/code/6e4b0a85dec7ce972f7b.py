import time
import math
import random
from collections import deque

def construct():
    start_time = time.time()
    DEADLINE = 110.0

    def sieve(n):
        is_p = bytearray(b'\x01') * (n + 1)
        is_p[0:2] = b'\x00\x00'
        for i in range(2, int(n**0.5) + 1):
            if is_p[i]:
                is_p[i*i::i] = b'\x00' * len(is_p[i*i::i])
        return [i for i in range(n + 1) if is_p[i]]

    small_primes = sieve(10000)

    def phi_fast(n):
        if n == 1:
            return 1
        result = n
        y = n
        for p in small_primes:
            if p * p > y:
                break
            if y % p == 0:
                result -= result // p
                while y % p == 0:
                    y //= p
        if y > 1:
            result -= result // y
        return result

    best_k = 1
    best_s = 1

    def scan(start, end):
        nonlocal best_k, best_s
        left = start
        phi_pos = {}
        q = deque()
        for n in range(start, end + 1):
            if time.time() - start_time > DEADLINE:
                return False
            pv = phi_fast(n)
            if pv in phi_pos:
                prev = phi_pos[pv]
                while left <= prev:
                    ln, lpv = q.popleft()
                    del phi_pos[lpv]
                    left = ln + 1
            phi_pos[pv] = n
            q.append((n, pv))
            k = n - left + 1
            if k > best_k:
                best_k = k
                best_s = left
        return True

    pos = 1
    while pos < 10**12:
        if time.time() - start_time > DEADLINE - 5:
            break
        elapsed = time.time() - start_time
        if elapsed < 10:
            chunk = 50000
        elif elapsed < 30:
            chunk = 100000
        elif elapsed < 60:
            chunk = 200000
        else:
            chunk = 400000
        end = min(pos + chunk - 1, 10**12)
        if not scan(pos, end):
            break
        pos = end + 1

    return {'start': best_s, 'length': best_k}
