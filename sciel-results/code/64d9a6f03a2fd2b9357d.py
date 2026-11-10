def solve(n, time_limit):
    # ...
    start_solve = time.time()
    # ... inside backtrack:
    if time.time() - start_solve > time_limit:
        return False
    # ...
