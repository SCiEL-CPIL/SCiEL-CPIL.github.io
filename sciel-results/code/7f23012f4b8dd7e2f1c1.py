total = 0
for i in range(3):
  for j in range(3):
    for k in range(3):
      if bs[i]==2 and bs[j]==2 and bs[k]==2:
        total += 7
      else:
        total += bs[i]*bs[j]*bs[k]
