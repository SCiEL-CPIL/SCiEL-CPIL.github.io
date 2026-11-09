   def check_feasible_fast(selected_list, candidate):
       sel = sorted(selected_list + [candidate])
       k = len(sel)
       for i in range(k):
           a = sel[i]
           for j in range(i+1, k):
               b = sel[j]
               for k2 in range(j, k):
                   c = sel[k2]
                   if (b + c) % a == 0:
                       return False
       return True
   