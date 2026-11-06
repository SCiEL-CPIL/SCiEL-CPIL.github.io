   if sp_out == sp_in:
       pass
   else:
       counts[sp_out] -= 1
       if counts[sp_out] == 0:
           del counts[sp_out]
       counts[sp_in] += 1
   