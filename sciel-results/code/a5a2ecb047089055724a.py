<<<<<<< SEARCH
import time
import math

def construct(n):
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    # Generate classic superpermutation
    word = generate_classic(n)
    
    target_count = math.factorial(n)
    
    # Local search with bitset tracking
    word = local_search_optimize(word, n, target_count, start_time, deadline)
    
    return "".join(map(str, word))
=======
import time
import math
import random

def construct(n):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Generate classic superpermutation
    word = generate_classic(n)
    
    target_count = math.factorial(n)
    
    # Local search with bitset tracking
    word = local_search_optimize(word, n, target_count, start_time, deadline)
    
    return "".join(map(str, word))
>>>>>>> REPLACE

