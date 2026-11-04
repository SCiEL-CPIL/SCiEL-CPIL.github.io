<<<<<<< SEARCH
import time
import random

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110
    
    # 1. Generate LUW graph as fallback
    n_luw, edges_luw = _luw_graph(k, g)
    
    # 2. Try to find a smaller graph using voltage assignment on K_4
    if k == 3 and g == 16:
        # Target n=936, Base K_4 (4 vertices), Group Z_234
        result = _search_k4_voltage(k, g, 234, deadline)
        if result:
            n, edges = result
            if n < n_luw:
                return n, edges
                
    return n_luw, edges_luw
=======
import time
import random
import math

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110
    
    # 1. Generate LUW graph as fallback
    n_luw, edges_luw = _luw_graph(k, g)
    
    # 2. Try to find a smaller graph using voltage assignment on Petersen graph
    if k == 3 and g == 16:
        # Petersen graph: 10 vertices, girth 5
        # n = 10 * m
        # Try various m values
        for m in [100, 120, 150, 180, 200, 250, 300, 400, 500, 600]:
            if time.time() > deadline - 5:
                break
            result = _search_petersen_voltage(k, g, m, deadline)
            if result:
                n, edges = result
                if n < n_luw:
                    return n, edges
                    
    return n_luw, edges_luw
>>>>>>> REPLACE

