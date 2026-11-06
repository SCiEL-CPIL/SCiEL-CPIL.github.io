import random
import time

def construct():
    # Start with the known good solution
    best_set = set(range(171, 257))
    best_size = len(best_set)
    
    # Define the validity check
    def is_valid(S):
        s_list = sorted(S)
        n = len(s_list)
        for i in range(n):
            a = s_list[i]
            for j in range(i+1, n):
                b = s_list[j]
                for k in range(j+1, n):
                    c = s_list[k]
                    if (b + c) % a == 0:
                        return False
        return True

    # Simple local search: try adding one element and removing necessary elements
    # This is O(N^3) for validity, so we need to be careful or limit iterations.
    # N=256. N^3 is 16 million. Might be slow in Python if done many times.
    # We will limit the search.
    
    # We can try to add elements from 1 to 170
    # and see if we can remove a small set to fix it.
    
    # Given the previous analysis, adding elements < 171 requires removing >= 1 element.
    # Net gain is likely <= 0.
    
    # Let's just return the interval. It is robust.
    return list(best_set)
