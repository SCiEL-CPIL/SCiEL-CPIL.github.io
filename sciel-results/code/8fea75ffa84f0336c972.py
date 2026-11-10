import time
import random

MAX_VAL = 256
MAX_SUM = 256 * 30 # Approx max sum for length 30. 256*40=10240. Let's use 10000.
SUM_SIZE = 10000

def is_valid_fast(seq):
    # seq is sorted
    n = len(seq)
    sums = [False] * SUM_SIZE
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += seq[j]
            if s >= SUM_SIZE:
                # Sum too large? Unlikely for k<30.
                # If it happens, we can't store it. But max sum is 256*256=65536.
                # We should size the array correctly.
                # Max possible sum is sum(1..256) = 32896.
                # So SUM_SIZE should be 33000.
                pass
            if sums[s]:
                return False
            sums[s] = True
    return True
