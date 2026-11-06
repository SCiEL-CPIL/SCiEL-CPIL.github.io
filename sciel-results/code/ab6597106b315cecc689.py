import time

def construct():
    start_time = time.time()
    DEADLINE = 82  # safety margin before 90s kill

    MAX_N = 1000000
    MAX_VAL = MAX_N + 24
    PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]

    # Compute rough part: i with all small prime factors removed
    rough = list(range(MAX_VAL + 1))
    for p in PRIMES:
        for i in range(p, MAX_VAL + 1, p):
            while rough[i] % p == 0:
                rough[i] //= p

    # smooth_value[i] = i // rough[i] = the 23-smooth part of i
    smooth_value = [0] * (MAX_VAL + 1)
    for i in range(1, MAX_VAL + 1):
        smooth_value[i] = i // rough[i]

    # Map distinct smooth values to compact integer IDs for fast freq array
    # Only values in the window range matter, but simplest: map all that appear
    val_to_id = {}
    sv_id = [0] * (MAX_VAL + 1)
    next_id = 0
    for i in range(1, MAX_VAL + 1):
        v = smooth_value[i]
        vid = val_to_id.get(v)
        if vid is None:
            vid = next_id
            val_to_id[v] = vid
            next_id += 1
        sv_id[i] = vid
    num_ids = next_id

    # Sliding window search for n in [1000, MAX_N]
    # Window for n: sv_id[n+1] .. sv_id[n+24]
    best_n = 1000
    best_count = 25

    # Initialize window for n=1000: indices 1001..1024
    freq = [0] * num_ids
    distinct = 0
    for i in range(1001, 1025):
        vid = sv_id[i]
        if freq[vid] == 0:
            distinct += 1
        freq[vid] += 1
    if distinct < best_count:
        best_count = distinct
        best_n = 1000

    for n in range(1001, MAX_N + 1):
        if time.time() - start_time > DEADLINE:
            break
        # Transition from n-1 to n: remove sv_id[n], add sv_id[n+24]
        out_id = sv_id[n]
        freq[out_id] -= 1
        if freq[out_id] == 0:
            distinct -= 1

        in_id = sv_id[n + 24]
        if freq[in_id] == 0:
            distinct += 1
        freq[in_id] += 1

        if distinct < best_count:
            best_count = distinct
            best_n = n
            if best_count == 1:
                break  # optimal, can't do better

    return best_n
