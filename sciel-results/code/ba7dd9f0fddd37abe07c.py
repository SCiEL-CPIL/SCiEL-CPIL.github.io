<<<<<<< SEARCH
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    n = 14
    m = 35
    max_val = 10000
    
    # Candidate value sets
    # 1. Consecutive small integers
    v1 = list(range(1, 15))
    # 2. Even numbers
    v2 = [2 * i for i in range(1, 15)]
    # 3. Multiples of 3
    v3 = [3 * i for i in range(1, 15)]
    # 4. Highly composite / small primes mix
    v4 = [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36]
    # 5. Another mix
    v5 = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20]
    # 6. Random small numbers
    random.seed(42)
    v6 = sorted(random.sample(range(1, 30), 14))
    v7 = sorted(random.sample(range(1, 40), 14))
    
    candidates = [v1, v2, v3, v4, v5, v6, v7]
=======
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    n = 14
    m = 35
    max_val = 10000
    
    # Candidate value sets - focus on multiplicative/additive collision-rich sets
    # 1. Consecutive small integers
    v1 = list(range(1, 15))
    # 2. Even numbers
    v2 = [2 * i for i in range(1, 15)]
    # 3. Multiples of 3
    v3 = [3 * i for i in range(1, 15)]
    # 4. Highly composite / small primes mix
    v4 = [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36]
    # 5. Another mix
    v5 = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20]
    # 6. Powers of 2 and 3 mixed
    v6 = [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36]
    # 7. Small composites only (many factor pairs)
    v7 = [4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22, 24]
    # 8. Multiples of 6 plus some
    v8 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16]
    # 9. Dense small range
    v9 = list(range(2, 16))
    # 10. Mix of small primes and composites
    v10 = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    # 11. Random small numbers
    random.seed(42)
    v11 = sorted(random.sample(range(1, 30), 14))
    v12 = sorted(random.sample(range(1, 40), 14))
    # 13. Very small range
    v13 = list(range(1, 15))
    # 14. Geometric-like progression
    v14 = [1, 2, 4, 8, 3, 6, 12, 24, 5, 10, 20, 40, 7, 14]
    v14 = sorted(v14)
    
    candidates = [v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14]
>>>>>>> REPLACE

