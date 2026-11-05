   for i in range(pw_len):
       a = pw_list[i]
       # Find max j such that pw_list[j] <= N_MAX - a
       # Since pw_list is sorted, we can use bisect_right
       limit = N_MAX - a
       max_j = bisect.bisect_right(pw_list, limit)
       for j in range(i, max_j):
           s = a + pw_list[j]
           counts[s] += 1
   